"""Modified training script with Tri-Oracle integration."""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any

import shardcast
import torch
import torch.distributed as dist
import torch.distributed.tensor
import wandb
from jaxtyping import Float
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from pydantic_config import parse_argv
from torch._guards import log as torch_log
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard  # type: ignore

from zeroband.training import envs
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state, save_ckpt_for_rollout
from zeroband.training.config import Config
from zeroband.training.data import BatchOutput, DatasetOutput, get_dataloader, packed_batch
from zeroband.training.loss import entropy_loss, grpo_loss, kl_penalty, selective_log_softmax
from zeroband.training.lr_scheduler import get_scheduler
from zeroband.training.utils import (
    MetricsAverager,
    PerfCounter,
    apply_ac_ckpt,
    log_prompt_response_samples,
    log_to_wandb,
    offload_model_to_cpu,
    reshard_module,
    wake_up_model_from_cpu,
)
from zeroband.training.world_info import WorldInfo, get_world_info
from zeroband.utils.http_monitor import HttpMonitor
from zeroband.utils.logger import get_logger
from zeroband.utils.models import ModelType, get_model_and_tokenizer

# Import Oracle integration
from zeroband.training.oracle_integration import create_oracle_integration, OracleConfig


def get_local_batch_size(batch_size: int, micro_bs: int, data_workers: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    assert batch_size % data_workers == 0, str(
        f"The batch size ({batch_size}) must be divisible by the number of data workers ({data_workers})."
    )

    return batch_size


def apply_fsdp(model: ModelType, reshard_after_forward: bool):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=layer_reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def get_device_placement(gpus_ids: list[int] | None, world_info: WorldInfo) -> int:
    """handle using a subset of GPUs. Should work like the CUDA_VISIBLE_DEVICES env var.
    The reason we use this is because in the rl launcher, torch is initialized before the env var is set, so we cannot use the CUDA_VISIBLE_DEVICES env var.
    """
    if gpus_ids is None:
        return world_info.local_rank


def get_logprobs(model: ModelType, input_ids: torch.Tensor, position_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    logits: Float[torch.Tensor, "batch seq vocab"] = model(input_ids=input_ids, position_ids=position_ids).logits.contiguous()

    input_ids_shifted = input_ids[:, 1:]
    logits_shifted = logits[:, :-1, :] / temperature
    logprobs = selective_log_softmax(logits_shifted, input_ids_shifted)
    del logits, logits_shifted
    return logprobs


def extract_code_from_completions(completions: List[str], tokenizer) -> List[str]:
    """Extract code from model completions."""
    codes = []
    for completion in completions:
        # Simple heuristic: look for code blocks or assume the whole completion is code
        # This should be adapted based on your prompt format
        if "```python" in completion:
            # Extract code between ```python and ```
            start = completion.find("```python") + len("```python")
            end = completion.find("```", start)
            if end > start:
                code = completion[start:end].strip()
            else:
                code = completion[start:].strip()
        elif "```" in completion:
            # Extract code between ``` and ```
            start = completion.find("```") + len("```")
            end = completion.find("```", start)
            if end > start:
                code = completion[start:end].strip()
            else:
                code = completion[start:].strip()
        else:
            # Assume the whole completion is code
            code = completion.strip()
        
        codes.append(code)
    
    return codes


def train_with_oracles(config: Config):
    if "ZERO_BAND_DEV" not in os.environ:
        torch_log.setLevel(logging.CRITICAL)

    logger = get_logger("TRAIN")
    world_info = get_world_info()
    wandb_sample_history = None

    if config.ckpt.clean_rollout_path and config.ckpt.rollout_path is not None:
        logger.info(f"Cleaning rollout path {config.ckpt.rollout_path}")
        shutil.rmtree(config.ckpt.rollout_path, ignore_errors=True)

    logger.info(f"start training on {world_info.world_size} rank(s)")

    # Initialize Oracle system
    oracle_config = OracleConfig(
        use_execution_oracle=getattr(config, "use_execution_oracle", True),
        use_static_oracle=getattr(config, "use_static_oracle", True),
        use_complexity_oracle=getattr(config, "use_complexity_oracle", True),
        use_documentation_oracle=getattr(config, "use_documentation_oracle", True),
        use_proof_oracle=getattr(config, "use_proof_oracle", False),
        use_reflective_oracle=getattr(config, "use_reflective_oracle", False),
        use_meta_gating=getattr(config, "use_meta_gating", True),
        execution_uncertainty_threshold=getattr(config, "execution_uncertainty_threshold", 0.3),
        device="cuda"
    )
    
    # Get model hidden dim from config or use default
    model_hidden_dim = getattr(config, "model_hidden_dim", 768)
    oracle_integration = create_oracle_integration(
        model_hidden_dim=model_hidden_dim,
        **oracle_config.__dict__
    )

    # Oracle loss weight (how much oracle losses contribute to total loss)
    oracle_loss_weight = getattr(config, "oracle_loss_weight", 0.3)

    # Allow eager fallback during production so that training runs don't die if compile fails
    torch.set_float32_matmul_precision("high")
    with envs.set_optional_torch_compile_eager_fallback():
        torch.compile = torch.compile(dynamic=True, fullgraph=False)  # type: ignore

    local_batch_size = get_local_batch_size(config.batch_size, config.train.micro_bs, config.data.num_workers, world_info)

    # Setup model
    assert config.model_id is not None, "Model ID must be provided"
    model, tokenizer = get_model_and_tokenizer(config.model_id, config.rope_theta, use_fast=True)
    model = model.to("cuda")

    # Apply FSDP
    if world_info.world_size > 1:
        apply_fsdp(model, reshard_after_forward=config.reshard_after_forward)

    # Reference model for KL penalty
    tensors_offloaded_reference = None
    if config.kl_coef is not None:
        assert config.model_reference_id is not None, "Model reference ID must be provided"
        model_reference, _ = get_model_and_tokenizer(config.model_reference_id, config.rope_theta, use_fast=True)
        model_reference = model_reference.to("cuda")
        if world_info.world_size > 1:
            apply_fsdp(model_reference, reshard_after_forward=True)
        tensors_offloaded_reference = offload_model_to_cpu(model_reference)

    # Optimizer setup
    params_group = model.parameters()
    if oracle_integration.meta_gating_network is not None:
        # Include meta-gating network parameters in optimization
        params_group = list(model.parameters()) + list(oracle_integration.meta_gating_network.parameters())
    
    optimizer = torch.optim.AdamW(params_group, lr=config.lr, weight_decay=config.optim.weight_decay, eps=config.optim.eps)

    # Training state
    training_progress = TrainingProgress(
        step=0,
        step_per_rollout=config.optim.step_per_rollout,
        shard_offset=0,
        epoch=0,
        consumed_samples=0,
        perf_counter=PerfCounter(world_info),
    )

    # Load checkpoint if exists
    checkpoint_loaded = False
    if config.ckpt.resumee:
        checkpoint_loaded = load_checkpoint_fsdp_state(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            checkpoint_dir=config.ckpt.path,
            rank=world_info.rank,
            training_progress=training_progress,
        )

    # Initialize wandb
    if world_info.rank == 0 and config.wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=config.model_dump(),
        )

    # Get dataloader
    train_dataloader = get_dataloader(
        config.data.data_path,
        config.data.split,
        config.data.seq_length,
        local_batch_size,
        config.data.num_workers,
        prefetch_factor=2,
        world_info=world_info,
    )
    train_dataloader_iterator = iter(train_dataloader)

    logger.info("Starting training loop with Oracle integration")

    while True:
        time_start = time.time()

        total_time_data_loading = 0
        total_time_packing = 0
        total_time_oracles = 0

        # Pre-compute logprobs with the model before update
        with torch.no_grad():
            if config.kl_coef is not None:
                wake_up_model_from_cpu(model_reference, tensors_offloaded_reference)
                del tensors_offloaded_reference

            data: list[list[BatchOutput]] = []
            oracle_feedback_per_rollout: list[list[Dict[str, Any]]] = []

            for rollout_step in range(config.optim.step_per_rollout):
                logger.debug(f"start rollout step {rollout_step} / {config.optim.step_per_rollout}")
                time_data_loading = time.time()

                batch_rollout: list[DatasetOutput] = next(train_dataloader_iterator)
                time_data_loading = time.time() - time_data_loading
                total_time_data_loading += time_data_loading

                time_0 = time.time()

                batch_packed = packed_batch(
                    batch_rollout, config.data.seq_length, tokenizer.pad_token_id, config.train.micro_bs, config.collate_mode
                )
                num_grad_acc_steps = len(batch_packed)

                time_1 = time.time()
                total_time_packing += time_1 - time_0

                oracle_feedback_batch = []

                for grad_acc_step in range(num_grad_acc_steps):
                    batch = batch_packed[grad_acc_step]
                    logger.debug(f"log prob grad_acc_step {grad_acc_step} / {num_grad_acc_steps}, batch: {batch['input_ids'].shape}")

                    input_ids = batch["input_ids"].to("cuda")

                    per_token_logps = get_logprobs(model, input_ids, batch["position_ids"], config.temperature)

                    batch["logprobs"] = per_token_logps.to("cpu")

                    if config.kl_coef is not None:
                        logger.debug(f"kl grad_acc_step {grad_acc_step} / {num_grad_acc_steps}, batch: {batch['input_ids'].shape}")
                        per_token_logps_reference = get_logprobs(model_reference, input_ids, batch["position_ids"], config.temperature)
                        batch["ref_logprobs"] = per_token_logps_reference.to("cpu")

                    # Run oracle evaluation on completions (if available)
                    if "completions" in batch and batch["completions"]:
                        time_oracle_start = time.time()
                        
                        # Extract code from completions
                        prompts = batch.get("prompts", [""] * len(batch["completions"]))
                        codes = extract_code_from_completions(batch["completions"], tokenizer)
                        
                        # Get oracle feedback
                        feedback = oracle_integration.get_oracle_feedback_for_inference(prompts, codes)
                        oracle_feedback_batch.append(feedback)
                        
                        total_time_oracles += time.time() - time_oracle_start
                    else:
                        oracle_feedback_batch.append(None)

                data.append(batch_packed)
                oracle_feedback_per_rollout.append(oracle_feedback_batch)

            if config.kl_coef is not None:
                reshard_module(model_reference)
                tensors_offloaded_reference = offload_model_to_cpu(model_reference)

            logprobs_aware_iterator = iter(data)
            oracle_feedback_iterator = iter(oracle_feedback_per_rollout)

            time_logprob = time.time() - time_start
            logger.info(f"Time to compute logprobs: {time_logprob:.2f} seconds, Oracle time: {total_time_oracles:.2f} seconds")

        logger.debug("start training rollout")

        # Training loop
        for rollout_step in range(config.optim.step_per_rollout):
            logger.debug(f"training rollout step {rollout_step} / {config.optim.step_per_rollout}")
            metric_averager = MetricsAverager()
            loss_batch = torch.tensor(0.0, device="cuda")

            data_per_rollout = next(logprobs_aware_iterator)
            oracle_feedback_per_batch = next(oracle_feedback_iterator)
            num_grad_acc_steps = len(data_per_rollout)

            # Collect samples for WandB logging
            if world_info.rank == 0 and config.wandb:
                batch = data_per_rollout[0]
                try:
                    wandb_sample_history = log_prompt_response_samples(tokenizer, batch, training_progress.step, wandb_sample_history)
                except Exception as e:
                    logger.warning(f"Error logging samples to WandB: {e}")

            # Gradient accumulation steps
            for grad_acc_step in range(num_grad_acc_steps):
                logger.debug(f"training grad_acc_step {grad_acc_step} / {num_grad_acc_steps}")
                batch = data_per_rollout[grad_acc_step]

                input_ids = batch["input_ids"].to("cuda")
                max_tokens = input_ids.shape[0] * input_ids.shape[1]

                loss_mask = batch["loss_mask"]
                advantage = batch["advantages"].to(input_ids.device)
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # Forward pass to get hidden states
                    outputs = model(input_ids=input_ids, position_ids=batch["position_ids"], output_hidden_states=True)
                    logits = outputs.logits.contiguous()
                    hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

                    # Compute GRPO loss
                    input_ids_shifted = input_ids[:, 1:]
                    logits_shifted = logits[:, :-1, :] / config.temperature
                    
                    loss = grpo_loss(
                        per_token_logps=batch["logprobs"].to(input_ids.device),
                        logits_shifted=logits_shifted,
                        input_ids_shifted=input_ids_shifted,
                        advantages=advantage,
                        loss_mask=loss_mask.to(input_ids.device),
                        clip_coef=config.clip_coef,
                    )

                    # Add entropy loss
                    if config.entropy_coef is not None:
                        loss_entropy = entropy_loss(logits_shifted, loss_mask.to(input_ids.device), return_token_avg=False).mean()
                        loss = loss + config.entropy_coef * loss_entropy
                        metric_averager.add("entropy", loss_entropy.item())

                    # Add KL penalty
                    if config.kl_coef is not None:
                        loss_kl = kl_penalty(
                            per_token_logps=batch["logprobs"].to(input_ids.device),
                            ref_logprobs=batch["ref_logprobs"].to(input_ids.device),
                            input_ids_shifted=input_ids_shifted,
                            loss_mask=loss_mask.to(input_ids.device),
                        )
                        loss = loss + config.kl_coef * loss_kl
                        metric_averager.add("kl", loss_kl.item())

                    # Add Oracle losses if we have completions and feedback
                    if oracle_feedback_per_batch[grad_acc_step] is not None and "completions" in batch:
                        prompts = batch.get("prompts", [""] * len(batch["completions"]))
                        codes = extract_code_from_completions(batch["completions"], tokenizer)
                        
                        # Compute oracle losses with gradients
                        oracle_loss, oracle_metrics = oracle_integration.compute_oracle_losses(
                            prompts=prompts,
                            generated_codes=codes,
                            hidden_states=hidden_states,
                            uncertainty_scores=None  # Will be calculated internally
                        )
                        
                        # Add oracle loss to total loss
                        if oracle_loss is not None and oracle_loss.requires_grad:
                            loss = loss + oracle_loss_weight * oracle_loss
                            metric_averager.add("oracle_loss", oracle_loss.item())
                            
                            # Add individual oracle metrics
                            for key, value in oracle_metrics.items():
                                metric_averager.add(f"oracle/{key}", value)

                    loss = loss / num_grad_acc_steps

                loss.backward()
                loss_batch += loss.item()

                metric_averager.add("grpo_loss", loss.item())
                metric_averager.add("loss", loss.item())
                metric_averager.add("advantage", advantage.mean().item())
                metric_averager.add("reward", batch["rewards"].mean().item())

            # Optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
            
            # Also clip oracle network gradients if using meta-gating
            if oracle_integration.meta_gating_network is not None:
                oracle_grad_norm = torch.nn.utils.clip_grad_norm_(
                    oracle_integration.meta_gating_network.parameters(), 
                    config.optim.grad_clip
                )
                metric_averager.add("oracle_grad_norm", oracle_grad_norm.item())
            
            optimizer.step()
            optimizer.zero_grad()

            # Update learning rate
            if config.lr_schedule == "cosine":
                scheduler = get_scheduler(
                    config.lr_schedule,
                    optimizer,
                    num_warmup_steps=config.warmup_steps,
                    num_training_steps=config.total_steps,
                    min_lr=config.min_lr,
                    warmup_ratio=config.warmup_ratio,
                )
                scheduler.step()
                metric_averager.add("lr", scheduler.get_last_lr()[0])

            # Log metrics
            metric_averager.add("grad_norm", grad_norm.item())
            training_progress.step += 1
            training_progress.consumed_samples += local_batch_size * world_info.world_size

            # Log to wandb
            if world_info.rank == 0 and config.wandb and training_progress.step % config.log_interval == 0:
                log_to_wandb(
                    {
                        **metric_averager.metrics(),
                        "step": training_progress.step,
                        "consumed_samples": training_progress.consumed_samples,
                        "time_per_step": training_progress.perf_counter.get_perf_since_last_log(),
                    }
                )

            # Save checkpoint
            if training_progress.step % config.ckpt.interval == 0:
                save_checkpoint_fsdp_state(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler if config.lr_schedule == "cosine" else None,
                    checkpoint_dir=config.ckpt.path,
                    training_progress=training_progress,
                    rank=world_info.rank,
                )

            # Save rollout checkpoint
            if config.ckpt.rollout_path is not None and training_progress.step % config.ckpt.rollout_interval == 0:
                save_ckpt_for_rollout(
                    model=model,
                    checkpoint_dir=config.ckpt.rollout_path,
                    training_progress=training_progress,
                    world_info=world_info,
                )

            logger.info(
                f"Step: {training_progress.step}, Loss: {loss_batch:.4f}, "
                f"Time: {time.time() - time_start:.2f}s"
            )


if __name__ == "__main__":
    config = parse_argv(Config)
    train_with_oracles(config)
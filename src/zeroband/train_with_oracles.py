"""Modified training script with Tri-Oracle integration."""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import shardcast
import torch
import torch.distributed.tensor
from jaxtyping import Float
from pydantic_config import parse_argv
from torch._guards import log as torch_log
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard  # type: ignore

import wandb
from zeroband.training import envs
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state, save_ckpt_for_rollout
from zeroband.training.config import Config
from zeroband.training.data import BatchOutput, DatasetOutput, get_dataloader, packed_batch
from zeroband.training.loss import entropy_loss, grpo_loss, kl_penalty, selective_log_softmax
from zeroband.training.lr_scheduler import get_scheduler

# Import Oracle integration
from zeroband.training.oracle_integration import OracleConfig, create_oracle_integration
from zeroband.training.utils import (
    MetricsAverager,
    PerfCounter,
    log_prompt_response_samples,
    log_to_wandb,
    offload_model_to_cpu,
    reshard_module,
    wake_up_model_from_cpu,
)
from zeroband.training.world_info import WorldInfo, get_world_info
from zeroband.utils.logger import get_logger
from zeroband.utils.models import ModelType, get_model_and_tokenizer


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
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    torch.cuda.set_device(get_device_placement(config.gpus_ids, world_info))

    local_batch_size = get_local_batch_size(config.optim.batch_size, config.train.micro_bs, config.data.num_workers, world_info)

    # Setup model
    model, tokenizer = get_model_and_tokenizer(config.model_name, config.train.attn_impl)

    # Apply FSDP
    if world_info.world_size > 1:
        apply_fsdp(model, reshard_after_forward=config.train.reshard_after_forward)

    # Reference model for KL penalty
    tensors_offloaded_reference = None
    if config.kl_coef is not None:
        model_reference, _ = get_model_and_tokenizer(config.model_name, config.train.attn_impl)
        model_reference = model_reference.to("cuda")
        if world_info.world_size > 1:
            apply_fsdp(model_reference, reshard_after_forward=True)
        tensors_offloaded_reference = offload_model_to_cpu(model_reference)

    # Optimizer setup
    params_group = model.parameters()
    if oracle_integration.meta_gating_network is not None:
        # Include meta-gating network parameters in optimization
        params_group = list(model.parameters()) + list(oracle_integration.meta_gating_network.parameters())

    optimizer = torch.optim.AdamW(params_group, lr=config.optim.optim.lr, weight_decay=config.optim.optim.weight_decay)

    # Training state
    training_progress = TrainingProgress(
        total_tokens=0,
        step=0,
        total_samples=0
    )

    # Performance counter
    perf_counter = PerfCounter(window_size=min(10, 2 * config.optim.step_per_rollout), model=model, seq_len=config.data.seq_length)

    # Load checkpoint if exists
    checkpoint_loaded = False
    if config.ckpt.resume is not None:
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
            project=config.project,
            entity=getattr(config, "wandb_entity", ""),
            name=config.wandb_run_name,
            config=config.model_dump(),
        )

    # Get dataloader
    train_dataloader, prefetcher = get_dataloader(
        tokenizer=tokenizer,
        local_batch_size=local_batch_size,
        batch_size=config.optim.batch_size * config.optim.step_per_rollout,
        data_config=config.data,
        step_count_init=0,
    )
    train_dataloader_iterator = iter(train_dataloader)

    logger.info("Starting training loop with Oracle integration")

    # Save initial rollout checkpoint at step 0 for inference to start
    if config.ckpt.rollout_path is not None:
        logger.debug("saving initial rollout ckpt at step 0")
        path = Path(config.ckpt.rollout_path) / "step_0"
        safetensor_path = save_ckpt_for_rollout(model, path)
        if world_info.rank == 0 and envs.SHARDCAST_OUTPUT_DIR is not None:
            logger.info(f"Broadcasting initial checkpoint {safetensor_path}")
            shardcast.broadcast(safetensor_path)

    iteration_count = 0
    while True:
        iteration_count += 1
        logger.info(f"Starting training iteration {iteration_count}")
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

                    # Run oracle evaluation on token sequences (convert to text first)
                    logger.debug(f"Batch keys: {list(batch.keys())}")
                    if "output_tokens" in batch and len(batch["output_tokens"]) > 0:
                        logger.info(f"Running oracle evaluation on {len(batch['output_tokens'])} outputs")
                        time_oracle_start = time.time()

                        # Convert token IDs to text
                        output_texts = []
                        input_texts = []
                        for i in range(len(batch["output_tokens"])):
                            # Decode output tokens
                            output_token_ids = batch["output_tokens"][i]
                            if hasattr(output_token_ids, 'tolist'):
                                output_token_ids = output_token_ids.tolist()
                            output_text = tokenizer.decode(output_token_ids, skip_special_tokens=True)
                            output_texts.append(output_text)
                            
                            # Decode input tokens for prompts
                            input_token_ids = batch["input_tokens"][i]
                            if hasattr(input_token_ids, 'tolist'):
                                input_token_ids = input_token_ids.tolist()
                            input_text = tokenizer.decode(input_token_ids, skip_special_tokens=True)
                            input_texts.append(input_text)

                        # Extract code from completions
                        codes = extract_code_from_completions(output_texts, tokenizer)

                        # Get oracle feedback
                        feedback = oracle_integration.get_oracle_feedback_for_inference(input_texts, codes)
                        oracle_feedback_batch.append(feedback)

                        total_time_oracles += time.time() - time_oracle_start
                        logger.info(f"Oracle evaluation completed in {time.time() - time_oracle_start:.2f}s")
                    else:
                        logger.debug("No output_tokens found in batch - skipping oracle evaluation")
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
                
                # Debug: Check advantage values
                if grad_acc_step == 0:  # Only log once per rollout
                    logger.info(f"Advantage stats: mean={advantage.mean().item():.4f}, std={advantage.std().item():.4f}, min={advantage.min().item():.4f}, max={advantage.max().item():.4f}")
                    if "rewards" in batch:
                        rewards = batch["rewards"]
                        logger.info(f"Reward stats: mean={rewards.mean().item():.4f}, std={rewards.std().item():.4f}, min={rewards.min().item():.4f}, max={rewards.max().item():.4f}")
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    # Forward pass to get hidden states
                    outputs = model(input_ids=input_ids, position_ids=batch["position_ids"], output_hidden_states=True)
                    logits = outputs.logits.contiguous()
                    hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

                    # Compute GRPO loss
                    pg_loss, clip_ratio = grpo_loss(
                        logits,
                        input_ids,
                        advantage,
                        batch["logprobs"].to(input_ids.device),
                        loss_mask.to(input_ids.device),
                        config.temperature,
                        getattr(config, 'grpo_epsilon_low', 1e-5),
                        getattr(config, 'grpo_epsilon_high', 1e5),
                        getattr(config, 'clamp_log_prob_coef', 0.3),
                        max_tokens,
                    )
                    loss = pg_loss

                    # Add entropy loss
                    entropy_coef = getattr(config, 'entropy_coef', None)
                    if entropy_coef is not None:
                        input_ids_shifted = input_ids[:, 1:]
                        logits_shifted = logits[:, :-1, :] / config.temperature
                        loss_entropy = entropy_loss(logits_shifted, loss_mask.to(input_ids.device), return_token_avg=False).mean()
                        loss = loss + entropy_coef * loss_entropy
                        metric_averager.update("entropy", loss_entropy)

                    # Add KL penalty
                    if config.kl_coef is not None:
                        if 'input_ids_shifted' not in locals():
                            input_ids_shifted = input_ids[:, 1:]
                        loss_kl = kl_penalty(
                            per_token_logps=batch["logprobs"].to(input_ids.device),
                            ref_logprobs=batch["ref_logprobs"].to(input_ids.device),
                            input_ids_shifted=input_ids_shifted,
                            loss_mask=loss_mask.to(input_ids.device),
                        )
                        loss = loss + config.kl_coef * loss_kl
                        metric_averager.update("kl", loss_kl)

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
                            metric_averager.update("oracle_loss", oracle_loss)

                            # Add individual oracle metrics
                            for key, value in oracle_metrics.items():
                                metric_averager.update(f"oracle/{key}", value)

                    loss = loss / num_grad_acc_steps

                loss.backward()
                loss_batch += loss.item()

                metric_averager.update("grpo_loss", loss)
                metric_averager.update("loss", loss)
                metric_averager.update("advantage", advantage.mean())
                metric_averager.update("reward", batch["rewards"].mean())

            # Optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config.optim, 'grad_clip', 1.0))

            # Also clip oracle network gradients if using meta-gating
            if oracle_integration.meta_gating_network is not None:
                oracle_grad_norm = torch.nn.utils.clip_grad_norm_(
                    oracle_integration.meta_gating_network.parameters(),
                    getattr(config.optim, 'grad_clip', 1.0)
                )
                metric_averager.update("oracle_grad_norm", oracle_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            # Update learning rate
            scheduler = None
            if getattr(config, 'lr_schedule', None) == "cosine":
                scheduler = get_scheduler(
                    getattr(config, 'lr_schedule', 'cosine'),
                    optimizer,
                    num_warmup_steps=getattr(config.optim, 'warmup_steps', 100),
                    num_training_steps=getattr(config.optim, 'total_steps', 10000),
                    min_lr=getattr(config, 'min_lr', 0),
                    warmup_ratio=getattr(config, 'warmup_ratio', 0.01),
                )
                scheduler.step()
                metric_averager.update("lr", scheduler.get_last_lr()[0])

            # Log metrics
            metric_averager.update("grad_norm", grad_norm)
            training_progress.step += 1
            training_progress.total_samples += local_batch_size * world_info.world_size

            # Log to wandb
            if world_info.rank == 0 and config.wandb and training_progress.step % getattr(config, 'log_interval', 50) == 0:
                log_to_wandb(
                    {
                        **metric_averager.metrics(),
                        "step": training_progress.step,
                        "total_samples": training_progress.total_samples,
                        "time_per_step": perf_counter.get_perf_since_last_log(),
                    }
                )

            # Save checkpoint
            if config.ckpt.interval is not None and training_progress.step % config.ckpt.interval == 0:
                save_checkpoint_fsdp_state(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler if getattr(config, 'lr_schedule', None) == "cosine" else None,
                    checkpoint_dir=config.ckpt.path,
                    training_progress=training_progress,
                    rank=world_info.rank,
                )

            # Save rollout checkpoint
            if config.ckpt.rollout_path is not None and training_progress.step % config.optim.step_per_rollout == 0:
                logger.debug("saving rollout ckpt")
                rollout_step = training_progress.step // config.optim.step_per_rollout
                path = Path(config.ckpt.rollout_path) / f"step_{rollout_step}"
                safetensor_path = save_ckpt_for_rollout(model, path)
                if world_info.rank == 0 and envs.SHARDCAST_OUTPUT_DIR is not None:
                    logger.info(f"Broadcasting {safetensor_path}")
                    shardcast.broadcast(safetensor_path)

            logger.info(
                f"Step: {training_progress.step}, Loss: {loss_batch:.4f}, "
                f"Time: {time.time() - time_start:.2f}s"
            )


if __name__ == "__main__":
    config = Config(**parse_argv())
    train_with_oracles(config)

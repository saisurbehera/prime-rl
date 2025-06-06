import json
import multiprocessing as mp
import os
import shutil
import time
import uuid
from pathlib import Path

# Import environment before any other imports
# ruff: noqa: I001
from zeroband.inference import envs

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
import torch.distributed as dist
from datasets import load_dataset
from pydantic_config import parse_argv
from toploc.utils import sha256sum
from vllm import LLM, SamplingParams

from zeroband.inference.config import Config
from zeroband.inference.parquet import get_parquet_table
from zeroband.inference.pipeline import setup_pipeline
from zeroband.inference.rewards import compute_rewards
from zeroband.inference.toploc import setup_toploc_cache
from zeroband.inference.utils import fake_chat_template, filter_data_by_prompt_length, generate_target_length_prompts, reload_model_weights
from zeroband.training.mp import EnvWrapper
from zeroband.utils.logger import get_logger
from zeroband.utils.metrics import PrimeMetric

# Global logger
logger = get_logger("INFER")


def inference(config: Config):
    # Initialize the logger
    logger.info("Starting inference")

    # Log relevant configuration
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Parallelism: TP={config.tp}, DP={config.dp}, PP={config.pp.world_size}")

    if config.clean_output_path and config.output_path is not None:
        logger.debug(f"Cleaning output path {config.output_path}")
        shutil.rmtree(config.output_path, ignore_errors=True)

    # Initialize metrics
    prime_metric = PrimeMetric(disable=config.prime_log_freq is None, period=config.prime_log_freq)

    # Initialize vLLM and get tokenizer
    logger.info(
        f"Initializing vLLM for {config.model_name} (max_model_len={config.max_model_len}, enforce_eager={config.enforce_eager}, dtype={config.dtype}, quant={config.quant})"
    )
    llm = LLM(
        model=config.model_name,
        tensor_parallel_size=config.tp,
        max_seq_len_to_capture=config.max_model_len,
        max_model_len=config.max_model_len,
        quantization=config.quant,
        enforce_eager=config.enforce_eager,
        disable_async_output_proc=True,  # We have an off by 1 error in toploc without this flag when cuda graph padding is enabled.
        download_dir=config.download_dir,
        dtype="bfloat16" if config.dtype == "bf16" else torch.float32,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(**config.sampling.model_dump())

    # Create communication for pipeline
    if config.pp.world_size > 1:
        setup_pipeline(
            llm=llm,
            rank=config.pp.rank,
            world_size=config.pp.world_size,
            iroh_seed=config.pp.iroh_seed,
            iroh_peer_id=config.pp.iroh_peer_id,
        )

    # Load  dataset
    dataset = load_dataset(config.dataset, split="train")
    logger.info(f"Loaded dataset {config.dataset} with {len(dataset):,} problems")

    # Optionally shuffle dataset
    if envs.GROUP_ID is not None:
        # We dont shuffle here because we shuffle reproducibly in the sampling loop.
        assert config.seed is None, "Seed is not supported when GROUP_ID is set"
        assert os.environ.get("DP_RANK") is None, "DP is not supported when GROUP_ID is set"
        node_address_int = int(envs.GROUP_ID, 16)
        logger.info(f"Seeding with {node_address_int} ({envs.GROUP_ID})")
    else:
        # Seed the dataset with a random number
        seed = config.seed + int(os.environ.get("DP_RANK", 0)) if config.seed is not None else None
        generator = np.random.default_rng(seed)
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(generator=generator)
        node_address_int = None

    if config.max_prompt_len:
        dataset = filter_data_by_prompt_length(dataset, config.max_prompt_len, tokenizer)
        logger.info(f"✨ Removed long prompts - {len(dataset)} samples remaining")

    # Optionally filter dataset
    if config.difficulty_filtering:
        logger.info(
            f"Filtering dataset for difficulty in [{config.difficulty_filtering.min_solve_rate}, {config.difficulty_filtering.max_solve_rate}]"
        )
        dataset = dataset.filter(
            lambda x: x[config.difficulty_filtering.solve_rate_field] >= config.difficulty_filtering.min_solve_rate
            and x[config.difficulty_filtering.solve_rate_field] <= config.difficulty_filtering.max_solve_rate
        )

    # Setup TOPLOC
    num_batch_samples = config.batch_size * config.sampling.n
    hidden_size = llm.llm_engine.model_executor.driver_worker.model_runner.model.config.hidden_size
    toploc_cache, _ = setup_toploc_cache(
        llm,
        disable=not config.toploc,
        max_seqs=num_batch_samples,
        hidden_size=hidden_size,
    )

    ckpt_step = 0
    real_step = 0
    if config.ckpt_start_path is not None:
        logger.info(f"Resuming from checkpoint {config.ckpt_start_path}")
        path = Path(config.ckpt_start_path)
        path_file = path / "model.safetensors"
        if not path_file.exists():
            raise FileNotFoundError(f"Checkpoint file {path_file} does not exist")
        ckpt_step = int(path.name.split("_")[-1])
        logger.info(f"Resuming from step {ckpt_step} at {path_file}")
        llm = reload_model_weights(llm, path_file)
        real_step = ckpt_step

    # This is used by the seeding logic to make sure we dont generate the same samples twice if we do multiple batches for a step
    current_step_batch_counter = 1
    total_problems = 0
    total_samples = 0
    total_tokens = 0
    max_samples = config.max_samples or len(dataset)

    for i in range(0, min(len(dataset), max_samples), config.batch_size):
        if config.step_endpoint is not None:
            # We get the step from the endpoint at the start of each batch to know what to work on
            try:
                new_real_step = requests.get(config.step_endpoint).json()
            except Exception as e:
                logger.warning(f"Failed to get step from endpoint {config.step_endpoint}: {e}")
                time.sleep(10)
                continue

            if new_real_step != real_step:
                real_step = new_real_step
                current_step_batch_counter = 1
            else:
                current_step_batch_counter += 1

        logger.info(f"Inference step {real_step} (Checkpoint step: {ckpt_step})")
        if config.rollout_path is not None and real_step - ckpt_step > config.async_level:
            logger.info(f"Required to reload model weights for step {ckpt_step} from {config.rollout_path}")
            ckpt_step = real_step - config.async_level
            attempt_count = 0
            while True:
                stable_file = Path(config.rollout_path) / f"step_{ckpt_step}/stable"
                if stable_file.exists():
                    logger.info(f"Reloading model weights for step {ckpt_step} from {stable_file}")
                    llm = reload_model_weights(llm, Path(config.rollout_path) / f"step_{ckpt_step}/model.safetensors")
                    total_problems = 0
                    total_tokens = 0
                    logger.info(f"Reloaded model weights for step {ckpt_step} from {stable_file}")
                    break
                if attempt_count % 30 == 0:
                    logger.info(f"No stable file found at {stable_file}, waiting for new checkpoint")
                time.sleep(1)
                attempt_count += 1

        # Get batch
        if node_address_int is not None:
            # TODO: What if we have multiple sample per real step?
            # Its impossible right now but we need to fix this if accept counter is used.

            # We reseed the generator here to make the sampling reproducible at each step.
            # This would work even if the node restarts and resumes from the current step.
            generator = np.random.default_rng(node_address_int * current_step_batch_counter + real_step)
            indices = generator.integers(0, len(dataset), config.batch_size)
        else:
            indices = list(range(i, min(i + config.batch_size, len(dataset))))

        logger.debug(f"Sampling batch with indices [{' '.join(map(str, indices[:3]))}...{' '.join(map(str, indices[-3:]))}]")
        batch = dataset.select(indices)

        messages = [[{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": "<think>\n"}] for item in batch]

        length_prompt_additions, target_lengths = generate_target_length_prompts(config.len_reward, len(batch))
        # Assume verification_info is stored as a JSON string in the dataset.
        verification_infos = [json.loads(item["verification_info"]) for item in batch]
        for target_length, verification_info in zip(target_lengths, verification_infos):
            verification_info["target_length"] = target_length
        task_types = [item["task_type"] for item in batch]

        if config.len_reward:
            if config.len_reward.length_prompt_location == "system_prompt":
                messages = [
                    [
                        {"role": "system", "content": length_prompt},
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": "<think>\n"},
                    ]
                    for item, length_prompt in zip(batch, length_prompt_additions)
                ]
            else:
                messages = [
                    [{"role": "user", "content": item["prompt"] + length_prompt}, {"role": "assistant", "content": "<think>\n"}]
                    for item, length_prompt in zip(batch, length_prompt_additions)
                ]
        else:
            messages = [
                [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": "<think>\n"}]
                for item, length_prompt in zip(batch, length_prompt_additions)
            ]

        if tokenizer.chat_template:
            prompts = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
            if config.model_name != "Qwen/QwQ-32B":
                for i, p in enumerate(prompts):
                    prompts[i] = p.replace("<｜begin▁of▁sentence｜>", "")
        else:
            prompts = fake_chat_template(messages)

        start_time = time.time()
        request_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        end_time = time.time()

        # Dropping like this isn't ideal. But in practice, we shouldn't have any prompts that are too long.
        request_outputs = [req for req in request_outputs if len(req.outputs[0].token_ids) > 0]
        if len(request_outputs) != len(prompts):
            logger.warning(f"{len(prompts) - len(request_outputs)} prompts were filtered out because they were too long")

        # This generates proofs for the remaining sequences that haven't reached max_len.
        # We call here to give time for the proofs to be generated non-blocking in the background.
        toploc_cache.maybe_generate_proofs_in_background(force_generate=True)

        # Calculate batch problems, samples and tokens
        batch_problems = len(batch)
        batch_samples = sum(len(req.outputs) for req in request_outputs)
        batch_input_tokens = sum(len(req.prompt_token_ids) for req in request_outputs)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in request_outputs)
        batch_tokens = batch_input_tokens + batch_output_tokens
        # Calculate overall problems, samples and tokens
        total_tokens += batch_tokens
        total_problems += batch_problems
        total_samples += batch_samples
        logger.info(f"Generated {batch_samples} samples for {batch_problems} problems for step {real_step} in {end_time - start_time:.2f}s")

        # Compute batch throughput and average sequence length
        batch_throughput = batch_tokens / (end_time - start_time)
        avg_sequence_length = batch_tokens / num_batch_samples
        logger.info(
            f"Batch throughput: {batch_throughput:.2f} tok/sec ({batch_tokens} tokens in {end_time - start_time:.2f}s, avg seq len: {avg_sequence_length:.1f})"
        )

        # Compute proofs
        # Note (Jack): Currently, vllm guarantees that seq ids are in the same order as prompts passed to generate.
        # Generate always adds requests to the engine in the order of the prompts.
        # And returns them in the sequence they were added.
        toploc_cache.wait_for_proofs()
        proofs = [b"".join(proofs) for _, proofs in sorted(toploc_cache.proofs.items(), key=lambda x: x[0])]
        toploc_cache.reset_cache()

        # Compute rewards and advantages
        start = time.time()
        request_rewards = compute_rewards(request_outputs, verification_infos, task_types, config.len_reward)
        logger.info(f"Computed rewards and advantages in {time.time() - start:.2f}s")

        table = get_parquet_table(
            request_outputs,
            request_rewards,
            proofs,
            ckpt_step,
            target_lengths,
        )

        step_path = Path(config.output_path) / f"step_{real_step}"
        os.makedirs(step_path, exist_ok=True)
        pq_save_path = f"{step_path}/{uuid.uuid4()}.parquet"
        pq.write_table(table, pq_save_path)
        logger.info(f"Saved batch outputs to {pq_save_path}")

        file_sha = sha256sum(pq_save_path)
        prime_metric.log_prime({"file_sha": file_sha, "file_name": pq_save_path})

        metric = {"dashbord-progress/total": total_problems, f"dashbord-progress/{config.dataset}": total_tokens}
        prime_metric.log_prime(metric)

        real_step += 1

        if config.total_step is not None and real_step > config.total_step:
            logger.info(f"Reached total step {config.total_step}, stopping inference")
            break

    logger.info(f"Inference finished! Generated {total_samples} samples for {total_problems} problems")

    # Manually destroy vLLM process group to avoid warnings
    dist.destroy_process_group()


def main(config: Config) -> list[mp.Process]:
    processes = []
    import zeroband.inference.envs as envs

    if config.dp > 1:
        if config.tp == "auto":
            assert torch.cuda.device_count() % config.dp == 0, "Number of GPUs must be divisible by DP"
            config.tp = torch.cuda.device_count() // config.dp
        gpu_ids = envs.CUDA_VISIBLE_DEVICES
        gpu_ids_per_rank = [gpu_ids[i : i + config.tp] for i in range(0, len(gpu_ids), config.tp)]
        for rank, gpu_ids in enumerate(gpu_ids_per_rank):
            envs = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)), "DP_RANK": str(rank)}
            process = mp.Process(target=EnvWrapper(inference, envs), args=(config,))
            processes.append(process)
    else:
        if config.tp == "auto":
            config.tp = torch.cuda.device_count()
        inference(config)

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == "__main__":
    # Set spawn method before any other multiprocessing code
    mp.set_start_method("spawn")
    config = Config(**parse_argv())  # type: ignore

    if config.step_endpoint is not None:
        current_step = requests.get(config.step_endpoint).json()
        assert isinstance(current_step, int), "Current step must be an integer"

    # Maybe start shardcast downloader
    from zeroband.inference import envs as inference_envs

    if inference_envs.SHARDCAST_SERVERS is not None:
        from zeroband.inference.shardcast_downloader import run_main_bg

        shardcast_process = run_main_bg(
            inference_envs.SHARDCAST_SERVERS,
            config.rollout_path,
            config.async_level + 1,
            # TODO: maybe +1 because we most likely won't download the current step in time?
            # We could deadlock though.
            max(current_step - config.async_level, 1),
        )
    else:
        shardcast_process = None

    try:
        main(config)

    finally:
        if shardcast_process is not None:
            import os
            import signal

            # SIGTERM is not working, so we use SIGKILL
            os.kill(shardcast_process.pid, signal.SIGKILL)
            shardcast_process.join()

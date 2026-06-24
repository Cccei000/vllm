"""Auto-tuning benchmark for DeepSeek V4 MHC TileLang kernels."""

import os
import argparse
import json
import time
from typing import Any
from collections import defaultdict
from tqdm import tqdm

import tilelang.language as T
import tilelang
import torch

from vllm.platforms import current_platform
import kernels

tilelang.disable_cache()

# Global pass configs
pass_configs = {
	tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
	tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
	tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10
}
ENABLE_PDL = current_platform.is_arch_support_pdl() and current_platform.is_cuda()


def load_model_config(config_path: str) -> dict[str, Any]:
	"""Load DeepSeek V4 config from JSON file."""
	with open(config_path, 'r') as f:
		config_dict = json.load(f)

	hc_mult = config_dict.get('hc_mult', 4)
	hidden_size = config_dict.get('hidden_size', 7168)

	return {
		'hc_mult': hc_mult,
		'hidden_size': hidden_size,
		'hc_hidden_size': hc_mult * hidden_size,
		'hc_mult3': hc_mult * (2 + hc_mult),  # hc_mult * 2 + hc_mult^2
		'n_out': hc_mult * (2 + hc_mult),  # Same as hc_mult3
	}


def get_gpu_info() -> dict[str, Any]:
	"""Get GPU information."""
	if not torch.cuda.is_available():
		return {'available': False}

	device_props = torch.cuda.get_device_properties(0)
	return {
		'available': True,
		'name': device_props.name,
		'memory_mb': device_props.total_memory // (1024 * 1024),
		'multi_processor_count': device_props.multi_processor_count,
	}


def benchmark_kernel(
	kernel_factory,
	configs,
	heuristics,
	**kwargs
) -> dict:
	"""Benchmark a single kernel with autotuning."""

	print(f"Search space: {len(configs)} configs")

	start_time = time.time()
	artifact = autotune_single_kernel(
		kernel_factory,
		configs,
		**kwargs
	)
	search_time = time.time() - start_time
	print(f"Best: {artifact.config}, latency: {artifact.latency:.4f}ms, time: {search_time:.1f}s")

	result = {'opt_config': artifact.config, 'opt_latency': artifact.latency}

	if heuristics:
		kwargs['use_pipeline'] = False
		kwargs['bench_multi_gpu'] = False
		heuristic_artifact = autotune_single_kernel(
			kernel_factory,
			heuristics,
			**kwargs
		)
		result['heuristic_config'] = heuristic_artifact.config
		result['heuristic_latency'] = heuristic_artifact.latency
		print(f"Heuristic: {heuristics[0]}")
		print(f"Heuristic latency: {heuristic_artifact.latency:.4f}ms")

	return result



def autotune_single_kernel(
	kernel_factory,
	configs,
	pass_configs=pass_configs,
	**kwargs
):
	"""Autotune a single kernel with given configs.

	Args:
		kernel_factory: Factory function that takes config and returns kernel
		configs: List of config dicts to search
		pass_configs: TileLang pass configs
		warmup: Number of warmup iterations
		rep: Number of profiling repetitions
		timeout: Timeout in seconds per config
		use_cudagraph: Whether to use CUDA graph backend for profiling

	Returns:
		Tuning artifact with best config and latency
	"""
	tuner = tilelang.autotuner.AutoTuner.from_kernel(kernel_factory, configs=configs)
	tuner = tuner.set_compile_args(
		target="auto",
		execution_backend="auto",
		verbose=False,
		pass_configs=pass_configs
	)

	# Set profile backend
	backend = 'cudagraph' if kwargs['use_cudagraph'] else 'event'
	tuner = tuner.set_profile_args(
		supply_type=tilelang.TensorSupplyType.Auto,
		ref_prog=None,
		skip_check=True,
		cache_input_tensors=False,
		backend=backend,
	)

	artifact = tuner.run(
    	warmup=kwargs['warmup'], 
     	rep=kwargs['rep'], 
      	timeout=kwargs['timeout'],
		use_pipeline=kwargs['use_pipeline'],
		benchmark_multi_gpu=kwargs['bench_multi_gpu']
    )
	return artifact


def run_autotune_benchmark(
	model_config: dict,
	num_tokens_list: list,
	n_splits_list: list,
	bench_heuristics: bool,
	**kwargs
) -> dict:
	"""Run autotune benchmark for all kernels.

	Returns results dict with structure:
	{
		num_tokens_value: {
			kernel_name: {
				"opt_config": {...},
				"opt_latency": float,
				"heuristic_config": {...},  # if bench_heuristics
				"heuristic_latency": float   # if bench_heuristics
			}
		}
	}
	"""
	print(kwargs)
 
	hc_mult = model_config['hc_mult']
	hidden_size = model_config['hidden_size']
	hc_hidden_size = model_config['hc_hidden_size']
	n_out = model_config['n_out']

	results = defaultdict(lambda: defaultdict(dict))

	total_tasks = len(num_tokens_list) * (
		1 + 1 + 1 + len(n_splits_list) * 3
	)
 
	# Benchmark mhc_post_kernel, hc_prenorm_gemm_block_m_kernel
	pbar = tqdm(total=total_tasks)
	for nt in num_tokens_list:
		pbar.set_description_str(f"Benchmarking mhc_post_kernel, num_tokens={nt}")
		results[nt]['mhc_post_kernel'] = benchmark_kernel(
			kernel_factory=kernels.mhc_post_kernel_factory(nt, hidden_size, hc_mult, enable_pdl=ENABLE_PDL),
			configs=kernels.get_mhc_post_kernel_configs(),
			heuristics=kernels.get_mhc_post_kernel_configs(use_heuristics=True) if bench_heuristics else None,
			**kwargs
		)
		pbar.update()

		pbar.set_description_str(f"Benchmarking hc_prenorm_gemm_block_m_kernel, num_tokens={nt}")
		results[nt]['hc_prenorm_gemm_block_m_kernel'] = benchmark_kernel(
			kernel_factory=kernels.hc_prenorm_gemm_block_m_kernel_factory(nt, hidden_size, hc_mult, n_out, enable_pdl=ENABLE_PDL),
			configs=kernels.get_hc_prenorm_gemm_block_m_kernel_configs(),
			heuristics=kernels.get_hc_prenorm_gemm_block_m_kernel_configs(use_heuristics=True) if bench_heuristics else None,
			**kwargs
		)
		pbar.update()

		pbar.set_description_str(f"Benchmarking hc_head_fuse_kernel, num_tokens={nt}")
		results[nt]['hc_head_fuse_kernel'] = benchmark_kernel(
			kernel_factory=kernels.hc_head_fuse_kernel_factory(nt, hidden_size, hc_mult, enable_pdl=ENABLE_PDL),
			configs=kernels.get_hc_head_fuse_kernel_configs(),
			heuristics=kernels.get_hc_head_fuse_kernel_configs(use_heuristics=True) if bench_heuristics else None,
			**kwargs
		)
		pbar.update()



	# Benchmark hc_prenorm_gemm_kernel and mhc_fused_kernel with n_splits	
	for nt in num_tokens_list:
		for ns in n_splits_list:
			
			# Benchmark hc_prenorm_gemm_kernel
			pbar.set_description_str(f"Benchmarking hc_prenorm_gemm_kernel, n_splits={ns}, num_tokens={nt}")
			results[nt]['hc_prenorm_gemm_kernel'][ns] = benchmark_kernel(
				kernel_factory=kernels.hc_prenorm_gemm_kernel_factory(nt, hidden_size, ns, hc_mult, n_out, enable_pdl=ENABLE_PDL),
				configs=kernels.get_hc_prenorm_gemm_kernel_configs(),
				heuristics=kernels.get_hc_prenorm_gemm_kernel_configs(True, nt, hidden_size * hc_mult, ns) if bench_heuristics else None,
				**kwargs
			)
			pbar.update()

			# Benchmark mhc_fused_kernel
			pbar.set_description_str(f"Benchmarking mhc_fused_kernel, n_splits={ns}, num_tokens={nt}")
			results[nt]['mhc_fused_kernel'][ns] = benchmark_kernel(
				kernel_factory=kernels.mhc_fused_kernel_factory(nt, hidden_size, ns, hc_mult, n_out, enable_pdl=ENABLE_PDL),
				configs=kernels.get_mhc_fused_kernel_configs(),
				heuristics=kernels.get_mhc_fused_kernel_configs(True, nt) if bench_heuristics else None,
				**kwargs
			)
			pbar.update()

			# Benchmark mhc_pre_big_fuse_with_norm_kernel
			pbar.set_description_str(f"Benchmarking mhc_pre_big_fuse_with_norm_kernel, n_splits={ns}, num_tokens={nt}")
			results[nt]['mhc_pre_big_fuse_with_norm_kernel'][ns] = benchmark_kernel(
				kernel_factory=kernels.mhc_pre_big_fuse_with_norm_kernel_factory(nt, hidden_size, ns, hc_mult, n_out, enable_pdl=ENABLE_PDL),
				configs=kernels.get_mhc_pre_big_fuse_with_norm_kernel_configs(),
				heuristics=kernels.get_mhc_pre_big_fuse_with_norm_kernel_configs(True) if bench_heuristics else None,
				**kwargs
			)
			pbar.update()




	print("\n" + "=" * 60)
	print("Benchmark completed!")
	print("=" * 60)

	return results


def main():
	"""Main entry point."""
	parser = argparse.ArgumentParser(
		description="Auto-tuning benchmark for DeepSeek V4 MHC TileLang kernels"
	)
	parser.add_argument(
		"--config",
		type=str,
		required=True,
		help="Path to DeepSeek V4 config.json",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="mhc_tuning_results.json",
		help="Output JSON file path (default: mhc_tuning_results.json)",
	)
	parser.add_argument(
		"--num-tokens",
		type=int,
		nargs="+",
		default=[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096],
		help="List of num_tokens to benchmark (default: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096)",
	)
	parser.add_argument(
		"--n-splits",
		type=int,
		nargs="+",
		default=[1],
		help="List of n_splits to benchmark (default: 1)",
	)
	parser.add_argument(
		"--bench-heuristics",
		action="store_true",
		default=False,
		help="Disable benchmarking heuristic configs",
	)
	parser.add_argument(
		"--use-cudagraph",
		action="store_true",
		default=False,
		help="Use CUDA graph backend for profiling (default: False)",
	)
	parser.add_argument(
		"--warmup",
		type=int,
		default=25,
		help="Number of warmup iterations (default: 25)",
	)
	parser.add_argument(
		"--rep",
		type=int,
		default=100,
		help="Number of profiling repetitions (default: 100)",
	)
	parser.add_argument(
		"--timeout",
		type=int,
		default=30,
		help="Timeout in seconds per config (default: 30)",
	)
	parser.add_argument(
		"--use-pipeline",
		action="store_true",
		default=False,
	)
	parser.add_argument(
		"--bench-multi-gpu",
		action="store_true",
		default=False,
	)

	args = parser.parse_args()

	# Load model config
	print(f"Loading config from: {args.config}")
	model_config = load_model_config(args.config)
	print(f"Model config: hc_mult={model_config['hc_mult']}, "
		  f"hidden_size={model_config['hidden_size']}, "
		  f"hc_hidden_size={model_config['hc_hidden_size']}, "
		  f"n_out={model_config['n_out']}")

	# Get GPU info
	gpu_info = get_gpu_info()
	if gpu_info['available']:
		print(f"GPU: {gpu_info['name']}, "
			  f"Memory: {gpu_info['memory_mb']}MB, "
			  f"SMs: {gpu_info['multi_processor_count']}")
	else:
		print("Error: CUDA not available!")
		return

	# Run benchmark
	results = run_autotune_benchmark(
		model_config=model_config,
		num_tokens_list=args.num_tokens,
		n_splits_list=args.n_splits,
		bench_heuristics=args.bench_heuristics,
		use_cudagraph=args.use_cudagraph,
		warmup=args.warmup,
		rep=args.rep,
		timeout=args.timeout,
		use_pipeline=args.use_pipeline,
		bench_multi_gpu=args.bench_multi_gpu
	)

	# Save results
	with open(args.output, "w") as f:
		json.dump(results, f, indent=4)
		f.write("\n")
	print(f"Results saved to: {args.output}")


if __name__ == "__main__":
	main()









import tilelang.language as T
import tilelang
import torch
import itertools
import math


def hc_prenorm_gemm_kernel_factory(num_tokens, hidden_size, n_splits, hc_mult, n_out, enable_pdl=True):
	
	def hc_prenorm_gemm_tilelang(n_thr, tile_n):
		# num_tokens = T.dynamic("num_tokens")
		hc_hidden_size = hc_mult * hidden_size
		k_per_split = hc_hidden_size // n_splits
		k_iters = k_per_split // n_thr
		n_tiles = T.ceildiv(n_out, tile_n)
		
		@T.prim_func
		def main(
			x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16),  # type: ignore[no-redef, valid-type]
			fn: T.Tensor((n_out, hc_hidden_size), T.float32),  # type: ignore[no-redef, valid-type]
			out: T.Tensor((n_splits, num_tokens, n_out), T.float32),  # type: ignore[no-redef, valid-type]
			sqrsum: T.Tensor((n_splits, num_tokens), T.float32),  # type: ignore[no-redef, valid-type]
		):
			with T.Kernel(num_tokens, n_tiles, n_splits, threads=n_thr) as (i_n, i_t, i_s,):
				tid = T.get_thread_binding()
				acc = T.alloc_local((tile_n,), T.float32)
				sqr = T.alloc_local((1,), T.float32)
				T.clear(acc)
				T.clear(sqr)

				if enable_pdl:
					T.pdl_sync()

				for it in T.serial(k_iters):
					i_k = i_s * k_per_split + it * n_thr + tid
					x_val = x[i_n, i_k]
					for i_o in T.unroll(tile_n):
						out_idx = i_t * tile_n + i_o
						if out_idx < n_out:
							acc[i_o] += x_val * fn[out_idx, i_k]
					if i_t == 0:
						sqr[0] += x_val * x_val

				for i_o in T.unroll(tile_n):
					acc[i_o] = T.warp_reduce_sum(acc[i_o])
				if i_t == 0:
					sqr[0] = T.warp_reduce_sum(sqr[0])

				lane = tid % 32
				warp_id = tid // 32
				num_warps = n_thr // 32
				warp_acc = T.alloc_shared((num_warps, tile_n), T.float32)
				warp_sqr = T.alloc_shared(num_warps, T.float32)

				if lane == 0:
					for i_o in T.unroll(tile_n):
						warp_acc[warp_id, i_o] = acc[i_o]
					if i_t == 0:
						warp_sqr[warp_id] = sqr[0]
				T.sync_threads()

				if warp_id == 0:
					if lane < tile_n:
						reduced_acc = T.alloc_var(T.float32, init=0.0)
						for i_w in T.unroll(num_warps):
							reduced_acc += warp_acc[i_w, lane]
						out_idx = i_t * tile_n + lane
						if out_idx < n_out:
							out[i_s, i_n, out_idx] = reduced_acc
					if lane == 0 and i_t == 0:
						reduced_sqr = T.alloc_var(T.float32, init=0.0)
						for i_w in T.unroll(num_warps):
							reduced_sqr += warp_sqr[i_w]
						sqrsum[i_s, i_n] = reduced_sqr

				if enable_pdl:
					T.pdl_trigger()
		
		return main
	return hc_prenorm_gemm_tilelang

def get_hc_prenorm_gemm_kernel_configs(
	use_heuristics=False,
	num_tokens=None,
	hidden_size=None,
	n_splits=None,
):
	"""Get configs for hc_prenorm_gemm kernel.

	Heuristics from tilelang.py:
	- If n_splits==1, num_tokens<128, and (hc_mult*hidden_size)%1024==0: use n_thr=1024, tile_n=4
	- Otherwise: use n_thr=512, tile_n=12
	"""
	if use_heuristics:
		if n_splits == 1 and num_tokens < 128 and hidden_size % 1024 == 0:
			return [{"n_thr": 1024, "tile_n": 4}]
		else:
			return [{"n_thr": 512, "tile_n": 12}]

	n_thr_list = [64, 128, 256, 512, 1024]
	tile_n_list = [2, 4, 8, 12, 16, 32]
	configs = list(itertools.product(n_thr_list, tile_n_list))
	configs = [{"n_thr": c[0], "tile_n": c[1]} for c in configs]
	return configs



def hc_prenorm_gemm_block_m_kernel_factory(num_tokens, hidden_size, hc_mult, n_out, enable_pdl=True):

	def hc_prenorm_gemm_block_m_tilelang(n_thr, tile_n, block_m):
		# num_tokens = T.dynamic("num_tokens")
		hc_hidden_size = hc_mult * hidden_size
		k_iters = hc_hidden_size // n_thr
		n_tiles = T.ceildiv(n_out, tile_n)
		m_tiles = T.ceildiv(num_tokens, block_m)     
		
		@T.prim_func
		def main(
			x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16),  # type: ignore[no-redef, valid-type]
			fn: T.Tensor((n_out, hc_hidden_size), T.float32),  # type: ignore[no-redef, valid-type]
			out: T.Tensor((1, num_tokens, n_out), T.float32),  # type: ignore[no-redef, valid-type]
			sqrsum: T.Tensor((1, num_tokens), T.float32),  # type: ignore[no-redef, valid-type]
		):

			with T.Kernel(m_tiles, n_tiles, threads=n_thr) as (i_mt, i_t):
				tid = T.get_thread_binding()
				acc = T.alloc_local((block_m, tile_n), T.float32)
				sqr = T.alloc_local((block_m,), T.float32)
				T.clear(acc)
				T.clear(sqr)

				if enable_pdl:
					T.pdl_sync()

				for it in T.serial(k_iters):
					i_k = it * n_thr + tid
					fn_val = T.alloc_local((tile_n,), T.float32)
					for i_o in T.unroll(tile_n):
						out_idx = i_t * tile_n + i_o
						if out_idx < n_out:
							fn_val[i_o] = fn[out_idx, i_k]
						else:
							fn_val[i_o] = 0.0
					for i_m in T.unroll(block_m):
						token_idx = i_mt * block_m + i_m
						if token_idx < num_tokens:
							x_val = x[token_idx, i_k]
							for i_o in T.unroll(tile_n):
								acc[i_m, i_o] += x_val * fn_val[i_o]
							if i_t == 0:
								sqr[i_m] += x_val * x_val

				for i_m in T.unroll(block_m):
					for i_o in T.unroll(tile_n):
						acc[i_m, i_o] = T.warp_reduce_sum(acc[i_m, i_o])
					if i_t == 0:
						sqr[i_m] = T.warp_reduce_sum(sqr[i_m])

				lane = tid % 32
				warp_id = tid // 32
				num_warps = n_thr // 32
				warp_acc = T.alloc_shared((num_warps, block_m, tile_n), T.float32)
				warp_sqr = T.alloc_shared((num_warps, block_m), T.float32)

				if lane == 0:
					for i_m in T.unroll(block_m):
						for i_o in T.unroll(tile_n):
							warp_acc[warp_id, i_m, i_o] = acc[i_m, i_o]
						if i_t == 0:
							warp_sqr[warp_id, i_m] = sqr[i_m]
				T.sync_threads()

				if warp_id == 0:
					for i_m in T.unroll(block_m):
						token_idx = i_mt * block_m + i_m
						if token_idx < num_tokens:
							if lane < tile_n:
								reduced_acc = T.alloc_var(T.float32, init=0.0)
								for i_w in T.unroll(num_warps):
									reduced_acc += warp_acc[i_w, i_m, lane]
								out_idx = i_t * tile_n + lane
								if out_idx < n_out:
									out[0, token_idx, out_idx] = reduced_acc
							if lane == 0 and i_t == 0:
								reduced_sqr = T.alloc_var(T.float32, init=0.0)
								for i_w in T.unroll(num_warps):
									reduced_sqr += warp_sqr[i_w, i_m]
								sqrsum[0, token_idx] = reduced_sqr

				if enable_pdl:
					T.pdl_trigger()
		return main
	return hc_prenorm_gemm_block_m_tilelang

def get_hc_prenorm_gemm_block_m_kernel_configs(
	use_heuristics=False,
	num_tokens=None,
	hidden_size=None,
	n_splits=None,
):
	"""Get configs for hc_prenorm_gemm_block_m kernel.

	Heuristics: default is n_thr=512, tile_n=12, block_m=2
	"""
	if use_heuristics:
		return [{"n_thr": 512, "tile_n": 12, "block_m": 2}]

	n_thr_list = [64, 128, 256, 512, 1024]
	tile_n_list = [2, 4, 8, 12, 16, 32]
	block_m_list = [1, 2, 4, 8, 16]
	configs = list(itertools.product(n_thr_list, tile_n_list, block_m_list))
	configs = [{"n_thr": c[0], "tile_n": c[1], "block_m": c[2]} for c in configs]
	return configs



def mhc_fused_kernel_factory(num_tokens, hidden_size, n_splits, hc_mult, n_out, enable_pdl=True):

	def mhc_fused_tilelang(n_thr, tile_n):
		"""Fused mhc post-mapping + pre-norm GEMM FMA"""
		m = num_tokens
		h = hidden_size
		hc = hc_mult
		split_k = n_splits
		# h_blk = math.gcd(hidden, h_blk)
		h_per_split = h // split_k
		n_tiles = n_out // tile_n
		h_iters = h_per_split // n_thr
		num_warps = n_thr // 32

		@T.prim_func
		def main(
			comb_mix: T.Tensor((m, hc, hc), T.float32),  # type: ignore[no-redef, valid-type]
			residual_in: T.Tensor((m, hc, h), T.bfloat16),  # type: ignore[no-redef, valid-type]
			post_mix: T.Tensor((m, hc), T.float32),  # type: ignore[no-redef, valid-type]
			x_in: T.Tensor((m, h), T.bfloat16),  # type: ignore[no-redef, valid-type]
			weight_t: T.Tensor((n_out, hc, h), T.float32),  # type: ignore[no-redef, valid-type]
			yp_out: T.Tensor((split_k, m, n_out), T.float32),  # type: ignore[no-redef, valid-type]
			rp_out: T.Tensor((split_k, m), T.float32),  # type: ignore[no-redef, valid-type]
			residual_out: T.Tensor((m, hc, h), T.bfloat16),  # type: ignore[no-redef, valid-type]
		):
			with T.Kernel(m, n_tiles, split_k, threads=n_thr) as (i_n, i_nt, i_ks):
				tid = T.get_thread_binding()
				warp_id = tid // 32
				lane = tid % 32

				s_warp = T.alloc_shared((num_warps, tile_n + 1), T.float32)
				s_post = T.alloc_shared((hc,), T.float32)
				s_comb = T.alloc_shared((hc, hc), T.float32)

				pm = T.alloc_local((hc,), T.float32)
				cm = T.alloc_local((hc, hc), T.float32)
				acc = T.alloc_local((tile_n,), T.float32)
				sqr = T.alloc_local((1,), T.float32)
				new_r = T.alloc_local((hc,), T.float32)

				T.clear(acc)
				T.clear(sqr)
				h_split_start = i_ks * h_per_split

				if enable_pdl:
					T.pdl_sync()

				T.copy(post_mix[i_n, 0], s_post)
				T.copy(comb_mix[i_n, 0, 0], s_comb)

				for j in T.unroll(hc):
					pm[j] = s_post[j]
				for j in T.unroll(hc):
					for k in T.unroll(hc):
						cm[k, j] = s_comb[k, j]

				# Each thread owns h_iters elements of the k-split's h slice.
				for it in T.serial(h_iters):
					h_idx = h_split_start + it * n_thr + tid

					# Compute new residual from layer output and past residual
					for j in T.unroll(hc):
						new_r[j] = pm[j] * x_in[i_n, h_idx]
						for k in T.unroll(hc):
							new_r[j] += cm[k, j] * residual_in[i_n, k, h_idx]

					# populate residual_out and compute sqr sum
					if i_nt == 0:
						for j in T.unroll(hc):
							residual_out[i_n, j, h_idx] = new_r[j]
							sqr[0] += new_r[j] * new_r[j]

					# Per-thread FMA into acc[n]
					for n in T.unroll(tile_n):
						for j in T.unroll(hc):
							acc[n] += weight_t[i_nt * tile_n + n, j, h_idx] * new_r[j]

				for n in T.unroll(tile_n):
					acc[n] = T.warp_reduce_sum(acc[n])
				if i_nt == 0:
					sqr[0] = T.warp_reduce_sum(sqr[0])

				# Cross-warp reduce via shared mem
				if lane == 0:
					for n in T.unroll(tile_n):
						s_warp[warp_id, n] = acc[n]
					if i_nt == 0:
						s_warp[warp_id, tile_n] = sqr[0]
				T.sync_threads()

				# Warp 0 does the final cross-warp sum and writes outputs
				if warp_id == 0:
					if lane < tile_n:
						v = T.alloc_var(T.float32, init=0.0)
						for w in T.unroll(num_warps):
							v += s_warp[w, lane]
						yp_out[i_ks, i_n, i_nt * tile_n + lane] = v

					if i_nt == 0 and lane == 0:
						v2 = T.alloc_var(T.float32, init=0.0)
						for w in T.unroll(num_warps):
							v2 += s_warp[w, tile_n]
						rp_out[i_ks, i_n] = v2

				if enable_pdl:
					T.pdl_trigger()

		return main
	return mhc_fused_tilelang
			
def get_mhc_fused_kernel_configs(
	use_heuristics=False,
	num_tokens=None,
	hidden_size=None,
	n_splits=None,
):
	"""Get configs for mhc_fused kernel.

	Heuristics from tilelang.py:
	- If num_tokens <= 16: n_thr=256, tile_n=2 if num_tokens<8 else 3
	- Otherwise: n_thr=256, tile_n=1
	"""
	if use_heuristics:
		if num_tokens <= 16:
			return [{"n_thr": 256, "tile_n": 2 if num_tokens < 8 else 3}]
		else:
			return [{"n_thr": 256, "tile_n": 1}]

	n_thr_list = [64, 128, 256, 512, 1024]
	tile_n_list = [1, 2, 3, 4, 8, 12, 16, 32]
	configs = list(itertools.product(n_thr_list, tile_n_list))
	configs = [{"n_thr": c[0], "tile_n": c[1]} for c in configs]
	return configs

   
   
def mhc_post_kernel_factory(num_tokens, hidden_size, hc_mult, enable_pdl=True):

	def mhc_post_tilelang(n_thr, h_blk):
		n = num_tokens
		h = hidden_size
		hc = hc_mult
		h_blk_actual = math.gcd(hidden_size, h_blk)

		@T.prim_func
		def main(
			a: T.Tensor((n, hc, hc), T.float32),  # type: ignore[no-redef, valid-type]
			b: T.Tensor((n, hc, h), T.bfloat16),  # type: ignore[no-redef, valid-type]
			c: T.Tensor((n, hc), T.float32),  # type: ignore[no-redef, valid-type]
			d: T.Tensor((n, h), T.bfloat16),  # type: ignore[no-redef, valid-type]
			x: T.Tensor((n, hc, h), T.bfloat16),  # type: ignore[no-redef, valid-type]
		):
			with T.Kernel(n, threads=n_thr) as i_n:
				b_shared = T.alloc_shared((hc, h_blk_actual), T.bfloat16)
				d_shared = T.alloc_shared(h_blk_actual, T.bfloat16)

				x_local = T.alloc_fragment((hc, h_blk_actual), T.float32)
				b_local = T.alloc_fragment((hc, h_blk_actual), T.float32)
				d_local = T.alloc_fragment(h_blk_actual, T.float32)

				a_local = T.alloc_fragment((hc, hc), T.float32)
				c_local = T.alloc_fragment(hc, T.float32)
				if enable_pdl:
					T.pdl_sync()
				T.copy(a[i_n, 0, 0], a_local)
				T.copy(c[i_n, 0], c_local)

				for i0_h in T.Serial(T.ceildiv(h, h_blk_actual)):
					T.copy(b[i_n, 0, i0_h * h_blk_actual], b_shared)
					T.copy(d[i_n, i0_h * h_blk_actual], d_shared)

					T.copy(b_shared, b_local)
					T.copy(d_shared, d_local)
					for i_hco, i1_h in T.Parallel(hc, h_blk_actual):
						x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
						for i_hci in T.vectorized(hc):
							x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

					T.copy(x_local, x[i_n, 0, i0_h * h_blk_actual])
				if enable_pdl:
					T.pdl_trigger()

		return main
	return mhc_post_tilelang

def get_mhc_post_kernel_configs(
	use_heuristics=False,
	num_tokens=None,
	hidden_size=None,
	n_splits=None,
):
	"""Get configs for mhc_post kernel.

	Heuristics: default is n_thr=128, h_blk=1024
	Note: h_blk is adjusted via math.gcd(hidden_size, h_blk) in kernel
	"""
	if use_heuristics:
		return [{"n_thr": 128, "h_blk": 1024}]

	n_thr_list = [64, 128, 256, 512, 1024]
	h_blk_list = [256, 512, 1024, 2048]
	configs = list(itertools.product(n_thr_list, h_blk_list))
	configs = [{"n_thr": c[0], "h_blk": c[1]} for c in configs]
	return configs



def hc_head_fuse_kernel_factory(num_tokens, hidden_size, hc_mult, enable_pdl=True):
	
	def hc_head_fuse_tilelang(n_thr, h_blk, n_stages, rms_eps=1e-06, hc_eps=1e-06):
		"""Two-pass fused kernel for hc_head.

		Pass 1: accumulate per-token squared sum and hc_mult dot-products
				(projections onto fn rows) using cross-thread reducers.
		Pass 2: apply sigmoid-gated weighted sum of residual channels to output.

		Avoids materialising mixes / rsqrt / pre tensors to global memory.
		"""
		# num_tokens = T.dynamic("num_tokens")
		hc_dim = hc_mult * hidden_size
		h_block = math.gcd(h_blk, hidden_size)
		n_h = hidden_size // h_block

		@T.prim_func
		def main(
			residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16],  # type: ignore[no-redef,valid-type]
			fn: T.Tensor[[hc_mult, hc_dim], T.float32] , # type: ignore[no-redef,valid-type]
			hc_scale: T.Tensor[[1], T.float32],  # type: ignore[no-redef,valid-type]
			hc_base: T.Tensor[[hc_mult], T.float32],  # type: ignore[no-redef,valid-type]
			out: T.Tensor[[num_tokens, hidden_size], T.bfloat16],  # type: ignore[no-redef,valid-type]
		):
			with T.Kernel(num_tokens, threads=n_thr) as i:
				if enable_pdl:
					T.pdl_sync()

				# ------------------------------------------------------------------
				# Pass 1 – for each residual channel m_c and h_block:
				#   • accumulate squared sum (for RMS norm denominator)
				#   • accumulate hc_mult dot-products with fn rows
				# ------------------------------------------------------------------
				sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
				mixes_r = T.alloc_reducer((hc_mult,), T.float32, replication="all")
				T.fill(sqrsum_r, 0.0)
				T.fill(mixes_r, 0.0)

				for m_c in T.serial(hc_mult):
					for i_h in T.serial(n_h):
						x_local = T.alloc_fragment(h_block, T.float32)
						T.copy(residual[i, m_c, i_h * h_block], x_local)

						for k in T.Parallel(h_block):
							sqrsum_r[0] += x_local[k] * x_local[k]

						for m_m in T.unroll(hc_mult):
							fn_local = T.alloc_fragment(h_block, T.float32)
							T.copy(fn[m_m, m_c * hidden_size + i_h * h_block], fn_local)
							for k in T.Parallel(h_block):
								mixes_r[m_m] += x_local[k] * fn_local[k]

				T.finalize_reducer(sqrsum_r)
				T.finalize_reducer(mixes_r)

				# ------------------------------------------------------------------
				# Compute pre_mix = sigmoid(mix * rsqrt * scale + base) + eps
				# ------------------------------------------------------------------
				pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
				rsqrt_val = T.alloc_fragment(1, T.float32)
				rsqrt_val[0] = T.rsqrt(sqrsum_r[0] / hc_dim + rms_eps)
				for m in T.Parallel(hc_mult):
					pre_mix_shared[m] = (
						T.sigmoid(mixes_r[m] * rsqrt_val[0] * hc_scale[0] + hc_base[m]) + hc_eps
					)

				# ------------------------------------------------------------------
				# Pass 2 – apply_mix: pipelined weighted sum over residual channels
				# ------------------------------------------------------------------
				for i0_h in T.Pipelined(n_h, num_stages=n_stages):
					xs = T.alloc_shared((hc_mult, h_block), T.bfloat16)
					xl = T.alloc_fragment((hc_mult, h_block), T.float32)
					T.copy(residual[i, 0, i0_h * h_block], xs, disable_tma=True)
					T.copy(xs, xl)

					ol = T.alloc_fragment(h_block, T.float32)
					T.clear(ol)
					for i_hc in T.serial(hc_mult):
						pre = pre_mix_shared[i_hc]
						for i1_h in T.Parallel(h_block):
							ol[i1_h] += pre * xl[i_hc, i1_h]

					T.copy(ol, out[i, i0_h * h_block], disable_tma=True)

				if enable_pdl:
					T.pdl_trigger()
	 
		return main
	return hc_head_fuse_tilelang

def get_hc_head_fuse_kernel_configs(
	use_heuristics=False,
	num_tokens=None,
	hidden_size=None,
	n_splits=None,
):

	if use_heuristics:
		return [{"n_thr": 128, "h_blk": 1024, "n_stages": 2}]

	n_thr_list = [64, 128, 256, 512]
	h_blk_list = [256, 512, 1024]
	n_stages_list = [2, 3, 4]
	configs = list(itertools.product(n_thr_list, h_blk_list, n_stages_list))
	configs = [{"n_thr": c[0], "h_blk": c[1], "n_stages": c[2]} for c in configs]
	return configs



def mhc_pre_big_fuse_with_norm_kernel_factory(num_tokens, hidden_size, n_splits, hc_mult, n_out, enable_pdl=True):
	
	def mhc_pre_big_fuse_with_norm_tilelang(
		n_thr: int,
		h_blk: int,
		n_stages: int, 
		rms_eps: float = 1e-06,
		hc_pre_eps: float =  1e-6,
		hc_sinkhorn_eps: float = 1e-6,
		hc_post_mult_value: float = 2.0,
		sinkhorn_repeat: int = 20,
		norm_eps: float = 1e-06,
	):
		# num_tokens = T.dynamic("num_tokens")
		# hc_mult3 = hc_mult * (2 + hc_mult)
		hc_mult3 = n_out
		hidden_block = math.gcd(h_blk, hidden_size)

		@T.prim_func
		def main(
				gemm_out_mul: T.Tensor[[n_splits, num_tokens, n_out], T.float32],  # type: ignore[no-redef, valid-type]
				gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32],  # type: ignore[no-redef, valid-type]
				hc_scale: T.Tensor[[3], T.float32],  # type: ignore[no-redef, valid-type]
				hc_base: T.Tensor[[hc_mult3], T.float32],  # type: ignore[no-redef, valid-type]
				residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16],  # type: ignore[no-redef, valid-type]
				post_mix: T.Tensor[[num_tokens, hc_mult], T.float32],  # type: ignore[no-redef, valid-type]
				comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32],  # type: ignore[no-redef, valid-type]
				layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16],  # type: ignore[no-redef, valid-type]
				norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
		):
			with T.Kernel(num_tokens, threads=n_thr) as i:
				rms = T.alloc_fragment(1, T.float32)
				mixes = T.alloc_fragment(hc_mult3, T.float32)
				T.clear(mixes)
				rms[0] = 0

				if enable_pdl:
					T.pdl_sync()

				for i_split in T.serial(n_splits):
					rms[0] += gemm_out_sqrsum[i_split, i]
				rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
				for j in T.Parallel(hc_mult3):
					mixes[j] = 0
					for i_split in T.serial(n_splits):
						mixes[j] += gemm_out_mul[i_split, i, j]
					mixes[j] *= rms[0]
				mixes_shared = T.alloc_shared(hc_mult3, T.float32)
				T.copy(mixes, mixes_shared)

				if T.get_thread_binding() < 32:
					cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
					for j in T.Parallel(hc_mult):
						post_mix[i, j] = (
							T.sigmoid(
								mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
							)
							* hc_post_mult_value
						)
					for j, k in T.Parallel(hc_mult, hc_mult):
						cm[j, k] = (
							mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
							+ hc_base[j * hc_mult + k + hc_mult * 2]
						)

					row_sum = T.alloc_fragment(hc_mult, T.float32)
					col_sum = T.alloc_fragment(hc_mult, T.float32)

					row_max = T.alloc_fragment(hc_mult, T.float32)
					T.reduce_max(cm, row_max, dim=1)
					for j, k in T.Parallel(hc_mult, hc_mult):
						cm[j, k] = T.exp(cm[j, k] - row_max[j])
					T.reduce_sum(cm, row_sum, dim=1)
					for j, k in T.Parallel(hc_mult, hc_mult):
						cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

					T.reduce_sum(cm, col_sum, dim=0)
					for j, k in T.Parallel(hc_mult, hc_mult):
						cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

					for _ in T.serial(sinkhorn_repeat - 1):
						T.reduce_sum(cm, row_sum, dim=1)
						for j, k in T.Parallel(hc_mult, hc_mult):
							cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

						T.reduce_sum(cm, col_sum, dim=0)
						for j, k in T.Parallel(hc_mult, hc_mult):
							cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

					for j, k in T.Parallel(hc_mult, hc_mult):
						comb_mix[i, j * hc_mult + k] = cm[j, k]
				else:
					pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
					for j in T.Parallel(hc_mult):
						pre_mix_shared[j] = (
							T.sigmoid(
								mixes_shared[j] * hc_scale[0] + hc_base[j],
							)
							+ hc_pre_eps
						)

					# Pass 1: stash unnormalized weighted-sum output in shared memory
					# as bf16 (matches the rounding that RMSNorm would see) while
					# accumulating the per-position squared sum.
					output_shared = T.alloc_shared(hidden_size, T.bfloat16)
					sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
					T.clear(sumsq_per_pos)

					for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=n_stages):
						xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
						xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
						T.copy(residual[i, 0, i0_h * hidden_block], xs)
						T.copy(xs, xl)

						ol = T.alloc_fragment(hidden_block, T.float32)
						T.clear(ol)

						for i_hc in T.serial(hc_mult):
							pre = pre_mix_shared[i_hc]
							for i1_h in T.Parallel(hidden_block):
								ol[i1_h] += pre * xl[i_hc, i1_h]

						for i1_h in T.Parallel(hidden_block):
							sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
							output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

					sumsq = T.alloc_fragment(1, T.float32)
					T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
					rsqrt_norm = T.alloc_fragment(1, T.float32)
					rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

					# Pass 2: scale by rsqrt * norm_weight and write the result to HBM.
					for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=n_stages):
						w_shared = T.alloc_shared(hidden_block, T.bfloat16)
						w_local = T.alloc_fragment(hidden_block, T.float32)
						T.copy(norm_weight[i0_h * hidden_block], w_shared)
						T.copy(w_shared, w_local)

						ol = T.alloc_fragment(hidden_block, T.float32)
						for i1_h in T.Parallel(hidden_block):
							ol[i1_h] = (
								output_shared[i0_h * hidden_block + i1_h]
								* rsqrt_norm[0]
								* w_local[i1_h]
							)

						T.copy(ol, layer_input[i, i0_h * hidden_block])

				if enable_pdl:
					T.pdl_trigger()

		return main
	return mhc_pre_big_fuse_with_norm_tilelang

def get_mhc_pre_big_fuse_with_norm_kernel_configs(
	use_heuristics=False,
	num_tokens=None,
	hidden_size=None,
	n_splits=None,
):

	if use_heuristics:
		return [{"n_thr": 96, "h_blk": 1024, "n_stages": 2}]

	n_thr_list = [96, 160, 288, 544]
	h_blk_list = [256, 512, 1024]
	n_stages_list = [2, 3, 4]
	configs = list(itertools.product(n_thr_list, h_blk_list, n_stages_list))
	configs = [{"n_thr": c[0], "h_blk": c[1], "n_stages": c[2]} for c in configs]
	return configs







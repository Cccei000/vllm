# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Autotune configuration loader for TileLang MHC kernels.

Reads offline autotune results from a JSON file (path given by the
``VLLM_MHC_AUTOTUNE_RESULT`` environment variable) and exposes query
functions that return optimal kernel configs for a given ``num_tokens``.

When the environment variable is unset, every public function returns
``None`` and callers fall back to their existing heuristics.
"""

from __future__ import annotations

import json
import os
from typing import Any
from vllm.logger import init_logger

logger = init_logger(__name__)

AutotuneConfig = dict[str, Any]

_ENV_VAR = "VLLM_MHC_AUTOTUNE_RESULT"
_AUTOTUNE_DATA: dict[int, dict] | None = None


def _init_autotune_data() -> dict[int, dict] | None:
    path = os.environ.get(_ENV_VAR)
    if not path or not os.path.exists(path):
        logger.info("VLLM_MHC_AUTOTUNE_RESULT does not exist."
                    " Using default config for mhc tilelang kernels.")
        return None
    with open(path) as f:
        raw = json.load(f)
    logger.info(f"Using VLLM_MHC_AUTOTUNE_RESULT from {path}"
            " for optimized mhc tilelang kernels.")
    return {int(k): v for k, v in raw.items()}


_AUTOTUNE_DATA = _init_autotune_data()


def _load_autotune_data() -> dict[int, dict] | None:
    return _AUTOTUNE_DATA


def _nearest_num_tokens(num_tokens: int, available: list[int]) -> int:
    return min(available, key=lambda x: abs(x - num_tokens))


def _kernel_entry(
    data: dict[int, dict],
    num_tokens: int,
    kernel_name: str,
    n_splits: str | None = None,
) -> tuple[AutotuneConfig, float] | None:
    available = sorted(data.keys())
    nt = _nearest_num_tokens(num_tokens, available)
    entry = data[nt].get(kernel_name)
    if entry is None:
        return None
    if n_splits is not None:
        entry = entry.get(n_splits)
        if entry is None:
            return None
    return entry["opt_config"], entry["opt_latency"]


def _best_gemm_latency(
    data: dict[int, dict], num_tokens: int, n_splits: int
) -> float:
    best = float("inf")
    if n_splits == 1:
        result = _kernel_entry(data, num_tokens,
                               "hc_prenorm_gemm_block_m_kernel")
        if result is not None:
            best = min(best, result[1])
    result = _kernel_entry(data, num_tokens, "hc_prenorm_gemm_kernel",
                           str(n_splits))
    if result is not None:
        best = min(best, result[1])
    return best


# ── Public API ──────────────────────────────────────────────────────


def get_mhc_post_config(num_tokens: int) -> AutotuneConfig | None:
    """Isolated: optimal config for ``mhc_post_tilelang`` kernel."""
    data = _load_autotune_data()
    if data is None:
        return None
    result = _kernel_entry(data, num_tokens, "mhc_post_kernel")
    if result is None:
        return None
    return result[0]


def get_hc_head_fused_config(num_tokens: int) -> AutotuneConfig | None:
    """Isolated: optimal config for ``hc_head_fuse_tilelang`` kernel.

    Returns ``{"n_thr": ..., "h_blk": ..., "n_stages": ...}``.
    The ``n_stages`` field is compile-time metadata and should be ignored
    when calling the kernel (only ``n_thr`` and ``h_blk`` are used).
    """
    data = _load_autotune_data()
    if data is None:
        return None
    result = _kernel_entry(data, num_tokens, "hc_head_fuse_kernel")
    if result is None:
        return None
    return result[0]


def get_hc_prenorm_gemm_config(
    num_tokens: int, n_splits: int
) -> tuple[str, AutotuneConfig] | None:
    """Joint: pick the faster GEMM variant for the given *n_splits*.

    Returns ``("block_m", config)`` or ``("split_k", config)``.
    """
    data = _load_autotune_data()
    if data is None:
        return None

    best_kernel: str | None = None
    best_latency = float("inf")
    best_config: AutotuneConfig = {}

    if n_splits == 1:
        result = _kernel_entry(data, num_tokens,
                               "hc_prenorm_gemm_block_m_kernel")
        if result is not None:
            cfg, lat = result
            if lat < best_latency:
                best_kernel, best_latency, best_config = "block_m", lat, cfg

    result = _kernel_entry(data, num_tokens, "hc_prenorm_gemm_kernel",
                           str(n_splits))
    if result is not None:
        cfg, lat = result
        if lat < best_latency:
            best_kernel, best_latency, best_config = "split_k", lat, cfg

    if best_kernel is None:
        return None
    return best_kernel, best_config


def get_mhc_pre_config(num_tokens: int) -> dict | None:
    """Joint: choose *n_splits* minimising GEMM + big_fuse_with_norm latency.

    Returns ``{"n_splits": int, "big_fuse_n_thr": int | None,
              "big_fuse_h_blk": int | None, "big_fuse_n_stages": int | None}``.
    """
    data = _load_autotune_data()
    if data is None:
        return None

    available = sorted(data.keys())
    nt = _nearest_num_tokens(num_tokens, available)
    nt_data = data[nt]

    candidate_splits: set[int] = set()
    for ns_str in nt_data.get("hc_prenorm_gemm_kernel", {}):
        candidate_splits.add(int(ns_str))
    for ns_str in nt_data.get("mhc_pre_big_fuse_with_norm_kernel", {}):
        candidate_splits.add(int(ns_str))
    if "hc_prenorm_gemm_block_m_kernel" in nt_data:
        candidate_splits.add(1)

    if not candidate_splits:
        return None

    big_fuse_entry = nt_data.get("mhc_pre_big_fuse_with_norm_kernel")

    best_total = float("inf")
    best_result: dict | None = None

    for ns in sorted(candidate_splits):
        gemm_lat = _best_gemm_latency(data, num_tokens, ns)
        if gemm_lat == float("inf"):
            continue

        bf_lat = 0.0
        bf_n_thr: int | None = None
        bf_h_blk: int | None = None
        bf_n_stages: int | None = None
        if big_fuse_entry is not None:
            bf_entry = big_fuse_entry.get(str(ns))
            if bf_entry is not None:
                bf_lat = bf_entry["opt_latency"]
                cfg = bf_entry["opt_config"]
                bf_n_thr = cfg.get("n_thr")
                bf_h_blk = cfg.get("h_blk")
                bf_n_stages = cfg.get("n_stages")

        total = gemm_lat + bf_lat
        if total < best_total:
            best_total = total
            best_result = {
                "n_splits": ns,
                "big_fuse_n_thr": bf_n_thr,
                "big_fuse_h_blk": bf_h_blk,
                "big_fuse_n_stages": bf_n_stages,
            }

    return best_result


def get_mhc_fused_post_pre_config(num_tokens: int) -> dict | None:
    """Joint: pick best (path, n_splits) for ``mhc_fused_post_pre_tilelang``.

    Compares:
    * fused path: ``mhc_fused_kernel[ns] + big_fuse_with_norm[ns]``
    * separate path: ``mhc_post_kernel + best_GEMM[ns] + big_fuse_with_norm[ns]``

    Returns a dict with ``"use_fused"``, ``"n_splits"``, and path-specific
    kernel configs.
    """
    data = _load_autotune_data()
    if data is None:
        return None

    available = sorted(data.keys())
    nt = _nearest_num_tokens(num_tokens, available)
    nt_data = data[nt]

    fused_entry = nt_data.get("mhc_fused_kernel", {})
    big_fuse_entry = nt_data.get("mhc_pre_big_fuse_with_norm_kernel")

    candidate_splits: set[int] = set()
    for ns_str in fused_entry:
        candidate_splits.add(int(ns_str))
    for ns_str in nt_data.get("hc_prenorm_gemm_kernel", {}):
        candidate_splits.add(int(ns_str))
    for ns_str in nt_data.get("mhc_pre_big_fuse_with_norm_kernel", {}):
        candidate_splits.add(int(ns_str))
    if "hc_prenorm_gemm_block_m_kernel" in nt_data:
        candidate_splits.add(1)

    if not candidate_splits:
        return None

    post_result = _kernel_entry(data, num_tokens, "mhc_post_kernel")
    post_latency = post_result[1] if post_result else float("inf")
    post_config = post_result[0] if post_result else None

    best_total = float("inf")
    best_result: dict | None = None

    for ns in sorted(candidate_splits):
        bf_lat = 0.0
        bf_n_thr: int | None = None
        bf_h_blk: int | None = None
        bf_n_stages: int | None = None
        if big_fuse_entry is not None:
            bf_sub = big_fuse_entry.get(str(ns))
            if bf_sub is not None:
                bf_lat = bf_sub["opt_latency"]
                cfg = bf_sub["opt_config"]
                bf_n_thr = cfg.get("n_thr")
                bf_h_blk = cfg.get("h_blk")
                bf_n_stages = cfg.get("n_stages")

        # ── fused path ──
        fused_sub = fused_entry.get(str(ns))
        if fused_sub is not None:
            fused_total = fused_sub["opt_latency"] + bf_lat
            if fused_total < best_total:
                best_total = fused_total
                best_result = {
                    "use_fused": True,
                    "n_splits": ns,
                    "fused_config": fused_sub["opt_config"],
                    "big_fuse_n_thr": bf_n_thr,
                    "big_fuse_h_blk": bf_h_blk,
                    "big_fuse_n_stages": bf_n_stages,
                }

        # ── separate path ──
        gemm_lat = _best_gemm_latency(data, num_tokens, ns)
        if gemm_lat < float("inf") and post_latency < float("inf"):
            separate_total = post_latency + gemm_lat + bf_lat
            if separate_total < best_total:
                best_total = separate_total
                best_result = {
                    "use_fused": False,
                    "n_splits": ns,
                    "post_config": post_config,
                    "big_fuse_n_thr": bf_n_thr,
                    "big_fuse_h_blk": bf_h_blk,
                    "big_fuse_n_stages": bf_n_stages,
                }

    return best_result

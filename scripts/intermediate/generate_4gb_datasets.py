#!/usr/bin/env python3
"""Generate 4GB .f32 datasets from iter rank_0 for weights and gradients."""
import os, math, glob, sys, time
import numpy as np
import torch

rank0 = os.path.expanduser("~/run/datasets/iter/rank_0")
out_root = "dataset"  # relative to project root (run from CCBench/)
target_f32 = 4 * 1024 * 1024 * 1024 // 4  # 1073741824 elements (4GB)

def load_pt(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        for k, v in data.items():
            if hasattr(v, 'shape'):
                data = v
                break
    return data.float().numpy().reshape(-1)

def save_tiled_f32(arr, out_path, label):
    n = arr.size
    repeat = math.ceil(target_f32 / n)
    tiled = np.tile(arr, repeat)[:target_f32]
    tiled.tofile(out_path)
    mb = tiled.size * 4 / 1024 / 1024
    gb = mb / 1024
    print(f"  -> {out_path} ({mb:.0f}MB = {gb:.2f}GB), tiled from {n} elements x{repeat}", flush=True)

def process_weights():
    print("[weight] Loading weight shard files...", flush=True)
    files = sorted(glob.glob(os.path.join(rank0, "DP_AllGather_Input_weight_shard_*.pt")))
    print(f"  found {len(files)} shard files", flush=True)
    chunks = []
    for f in files:
        a = load_pt(f)
        chunks.append(a)
        print(".", end="", flush=True)
    print()
    combined = np.concatenate(chunks)
    print(f"  combined: {combined.size} f32 elements ({combined.size*4/1024/1024:.0f}MB)", flush=True)
    out_dir = os.path.join(out_root, "weight")
    os.makedirs(out_dir, exist_ok=True)
    save_tiled_f32(combined, os.path.join(out_dir, "weight_4gb.f32"), "weight")

def process_dp_grads():
    print("[dp_grad] Loading DP gradient shard files...", flush=True)
    files = sorted(glob.glob(os.path.join(rank0, "DP_ReduceScatter_grad_shard_*.pt")))
    print(f"  found {len(files)} shard files", flush=True)
    chunks = []
    for f in files:
        a = load_pt(f)
        chunks.append(a)
        print(".", end="", flush=True)
    print()
    combined = np.concatenate(chunks)
    print(f"  combined: {combined.size} f32 elements ({combined.size*4/1024/1024:.0f}MB)", flush=True)
    out_dir = os.path.join(out_root, "dp_grad")
    os.makedirs(out_dir, exist_ok=True)
    save_tiled_f32(combined, os.path.join(out_dir, "dp_grad_4gb.f32"), "dp_grad")

def process_gradients():
    """Combine bwd_grad + dp_grad for the generic 'gradients' dataset."""
    print("[gradients] Loading gradient files...", flush=True)
    arrs = []
    for f in ['PP_Comm_recv_bwd_grad.pt', 'SP_AllGather_bwd_grad_chunk_before.pt']:
        a = load_pt(os.path.join(rank0, f))
        print(f"  {f}: {a.size} elements", flush=True)
        arrs.append(a)
    # Also add dp_grad shards
    dp_files = sorted(glob.glob(os.path.join(rank0, "DP_ReduceScatter_grad_shard_*.pt")))
    for f in dp_files:
        a = load_pt(f)
        arrs.append(a)
    combined = np.concatenate(arrs)
    print(f"  combined: {combined.size} f32 elements ({combined.size*4/1024/1024:.0f}MB)", flush=True)
    out_dir = os.path.join(out_root, "gradients")
    os.makedirs(out_dir, exist_ok=True)
    save_tiled_f32(combined, os.path.join(out_dir, "grad_4gb.f32"), "gradients")

if __name__ == "__main__":
    t0 = time.time()
    process_weights()
    process_dp_grads()
    process_gradients()
    t1 = time.time()
    print(f"\nDone in {t1-t0:.1f}s", flush=True)

#!/usr/bin/env python3
"""Generate 512MB raw .f32 datasets from ~/run/datasets/iter/rank_0"""
import os, math, glob, sys
import numpy as np
import torch

rank0 = os.path.expanduser("~/run/datasets/iter/rank_0")
out_root = "dataset"  # relative to project root (run from CCBench/)
target_f32 = 512 * 1024 * 1024 // 4  # 134217728 elements

def load_pt(path):
    """Load a .pt file and return as flat float32 numpy array."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        for k, v in data.items():
            if hasattr(v, 'shape'):
                data = v
                break
    return data.float().numpy().reshape(-1)

def save_tiled_f32(arr, out_path):
    """Tile a 1D float32 array to 512MB, save as raw binary."""
    n = arr.size
    repeat = math.ceil(target_f32 / n)
    tiled = np.tile(arr, repeat)[:target_f32]
    tiled.tofile(out_path)
    return tiled.size * 4

def process_activations():
    print("[act] Loading activation files...", flush=True)
    arrs = []
    for f in ['SP_ReduceScatter_fwd_activation_global_before.pt',
              'PP_Comm_send_fwd_activation.pt']:
        a = load_pt(os.path.join(rank0, f))
        print(f"  {f}: {a.size} elements", flush=True)
        arrs.append(a)
    combined = np.concatenate(arrs)
    print(f"  combined: {combined.size} f32 elements", flush=True)
    out_dir = os.path.join(out_root, "activation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "act_cache_512mb.f32")
    size = save_tiled_f32(combined, out_path)
    print(f"  -> {out_path} ({size/1024/1024:.0f}MB)", flush=True)

def process_bwd_grads():
    print("[bwd] Loading backward gradient files...", flush=True)
    arrs = []
    for f in ['PP_Comm_recv_bwd_grad.pt',
              'SP_AllGather_bwd_grad_chunk_before.pt']:
        a = load_pt(os.path.join(rank0, f))
        print(f"  {f}: {a.size} elements", flush=True)
        arrs.append(a)
    combined = np.concatenate(arrs)
    print(f"  combined: {combined.size} f32 elements", flush=True)
    out_dir = os.path.join(out_root, "bwd_grad")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bwd_grad_512mb.f32")
    size = save_tiled_f32(combined, out_path)
    print(f"  -> {out_path} ({size/1024/1024:.0f}MB)", flush=True)

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
    print(f"  combined: {combined.size} f32 elements", flush=True)
    out_dir = os.path.join(out_root, "weight")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "weight_512mb.f32")
    size = save_tiled_f32(combined, out_path)
    print(f"  -> {out_path} ({size/1024/1024:.0f}MB)", flush=True)

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
    print(f"  combined: {combined.size} f32 elements", flush=True)
    out_dir = os.path.join(out_root, "dp_grad")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dp_grad_512mb.f32")
    size = save_tiled_f32(combined, out_path)
    print(f"  -> {out_path} ({size/1024/1024:.0f}MB)", flush=True)

if __name__ == "__main__":
    import time
    t0 = time.time()
    process_activations()
    process_bwd_grads()
    process_weights()
    process_dp_grads()
    t1 = time.time()
    print(f"\nDone in {t1-t0:.1f}s", flush=True)

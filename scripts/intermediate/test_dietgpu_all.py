#!/usr/bin/env python3
"""Measure dietgpu compression ratio across datasets at multiple sizes."""
import os, sys, time
import numpy as np
import torch

SO_PATH = "/data/run01/scyb672/fhz/mybench/plugin_projects/ccl/uccl/thirdparty/dietgpu/p2p_dietgpu.cpython-310-x86_64-linux-gnu.so"

DATASETS = {
    "gradients": "/data/run01/scyb672/fhz/mybench/dataset/gradients/grad_512mb.f32",
    "kv":        "/data/run01/scyb672/fhz/mybench/dataset/kv/kv_cache_512mb.f32",
    "activation": "/data/run01/scyb672/fhz/mybench/dataset/activation/act_cache_512mb.f32",
    "weight":    "/data/run01/scyb672/fhz/mybench/dataset/weight/weight_512mb.f32",
}

# From 8KB to 512MB in float32 elements
SIZES = [
    (8192,       "8KB"),
    (16384,      "16KB"),
    (131072,     "128KB"),
    (1048576,    "1MB"),
    (8388608,    "8MB"),
    (33554432,   "32MB"),
    (67108864,   "64MB"),
    (134217728,  "128MB"),
    (268435456,  "256MB"),
    (536870912,  "512MB"),
]

def main():
    torch.ops.load_library(SO_PATH)

    print(f"{'Dataset':<14} {'Size':<8} {'FloatCodec':<12} {'ANSCodec':<12} {'Float:MB':<10} {'ANS:MB':<10}")
    print(f"{'':-<14} {'':-<8} {'':-<12} {'':-<12} {'':-<10} {'':-<10}")

    for name, path in DATASETS.items():
        data = np.fromfile(path, dtype=np.float32)
        t_gpu_base = torch.from_numpy(data).cuda()

        for n_elems, size_label in SIZES:
            if n_elems > len(data):
                n_elems = len(data)
            t = t_gpu_base[:n_elems]
            orig = t.numel() * t.element_size()

            # Float codec
            torch.cuda.synchronize()
            comp_f, sizes_f, _ = torch.ops.dietgpu.compress_data(True, [t], False, None, None, None)
            torch.cuda.synchronize()
            sz_f = sizes_f.cpu()[0].item()
            ratio_f = sz_f / orig

            # ANS codec
            comp_a, sizes_a, _ = torch.ops.dietgpu.compress_data(False, [t], False, None, None, None)
            torch.cuda.synchronize()
            sz_a = sizes_a.cpu()[0].item()
            ratio_a = sz_a / orig

            print(f"{name:<14} {size_label:<8} {ratio_f:<12.4f} {ratio_a:<12.4f} {sz_f/1024/1024:<10.2f} {sz_a/1024/1024:<10.2f}")

            if n_elems == len(data):
                break  # no larger sizes to test for this dataset

    print(f"\n{'':=^70}")
    print(f"  Ratio < 1.0 = compression achieved. Lower is better.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

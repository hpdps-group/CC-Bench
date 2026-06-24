#!/usr/bin/env python3
"""Test dietgpu compression on a .f32 file. Reports compression ratio."""
import os, sys, time
import numpy as np
import torch

SO_PATH = "plugin_projects/ccl/uccl/thirdparty/dietgpu/p2p_dietgpu.cpython-310-x86_64-linux-gnu.so"

# Pick a dataset — small one for quick test
DS_DIR = "dataset"
TEST_FILE = os.path.join(DS_DIR, "kv/kv_cache_512mb.f32")

def main():
    # Load the extension
    torch.ops.load_library(SO_PATH)
    print(f"Loaded dietgpu extension from: {SO_PATH}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_properties(i).name}")

    # Load .f32 file
    print(f"\nLoading: {TEST_FILE}")
    t0 = time.time()
    data = np.fromfile(TEST_FILE, dtype=np.float32)
    t1 = time.time()
    raw_bytes = data.nbytes
    print(f"  {len(data)} elements, {raw_bytes/1024/1024:.1f} MB")
    print(f"  Load time: {t1-t0:.2f}s")

    # Move to GPU — use first N elements if too large (512 MB may be too big for GPU mem)
    max_elems = 64 * 1024 * 1024  # 256 MB
    if len(data) > max_elems:
        data = data[:max_elems]
        print(f"  Trimmed to {len(data)} elements ({data.nbytes/1024/1024:.1f} MB)")

    t0 = time.time()
    t = torch.from_numpy(data).cuda()
    t1 = time.time()
    print(f"  GPU transfer: {t1-t0:.2f}s")
    print(f"  GPU tensor shape: {t.shape}, dtype: {t.dtype}")

    # Compress
    print(f"\nCompressing with dietgpu float codec...")
    torch.cuda.synchronize()
    t0 = time.time()

    # Using compress_data (returns (compressed_tensor, sizes_tensor, temp_mem_usage))
    comp, sizes, temp_mem = torch.ops.dietgpu.compress_data(
        True,        # compress_as_float
        [t],         # list of tensors
        False,       # checksum
        None,        # temp_mem
        None,        # out_compressed
        None         # out_compressed_sizes
    )

    torch.cuda.synchronize()
    t1 = time.time()

    # Get compressed size
    size_host = sizes.cpu()
    comp_size = size_host[0].item()
    original_bytes = t.numel() * t.element_size()

    ratio = comp_size / original_bytes
    bpp = comp_size * 8 / t.numel()  # bits per element

    print(f"  Compress time: {t1-t0:.3f}s")
    print(f"  Original:     {original_bytes:>12} bytes  ({original_bytes/1024/1024:.2f} MB)")
    print(f"  Compressed:   {comp_size:>12} bytes  ({comp_size/1024/1024:.2f} MB)")
    print(f"  Temp mem:     {temp_mem/1024/1024:.1f} MB")
    print(f"  Ratio:        {ratio:.4f}x  ({ratio*100:.2f}% of original)")
    print(f"  BPP:          {bpp:.2f} bits/element (float32 = 32)")

    # Also test with ANS-only (byte-level) compression for comparison
    print(f"\nCompressing with dietgpu ANS-only codec (byte-level)...")
    torch.cuda.synchronize()
    t0_ans = time.time()

    comp_ans, sizes_ans, temp_mem_ans = torch.ops.dietgpu.compress_data(
        False,       # compress_as_float = False -> ANS-only
        [t],
        False,       # checksum
        None, None, None
    )

    torch.cuda.synchronize()
    t1_ans = time.time()

    size_host_ans = sizes_ans.cpu()
    comp_size_ans = size_host_ans[0].item()
    ratio_ans = comp_size_ans / original_bytes

    print(f"  Compress time: {t1_ans-t0_ans:.3f}s")
    print(f"  Compressed:   {comp_size_ans:>12} bytes  ({comp_size_ans/1024/1024:.2f} MB)")
    print(f"  Ratio:        {ratio_ans:.4f}x  ({ratio_ans*100:.2f}% of original)")
    print(f"  BPP:          {comp_size_ans*8/t.numel():.2f} bits/element")

    return 0

if __name__ == "__main__":
    sys.exit(main())

# configure-bench Auto-Configuration Skill

## Overview

This Skill guides the AI agent to automatically detect the CCBench project environment and, based on the detection results combined with user interaction, complete the generation and modification of all configuration files.

## Workflow (6 Steps)

```
Step 1: Environment Detection  ──►  Run detect_env.sh, parse JSON output
Step 2: Analysis & Display     ──►  Present environment summary to user
Step 3: Interactive Q&A        ──►  Ask questions with detection-based defaults
Step 4: Config Generation      ──►  Write / modify 8 JSONC config files
Step 5: Wrapper Generation     ──►  If needed, generate comm/compression wrappers (last, context-intensive)
Step 6: Summary & Confirm      ──►  Show full config, user confirms, print disclaimer
```

---

## Step 1: Environment Detection

Run the detection script (from the project root):

```bash
bash .claude/skills/configure-bench/scripts/detect_env.sh 2>/dev/null
```

The script outputs **structured JSON** with the following key fields:

| Field | Description | Used For |
|---|---|---|
| `hostname` | Host name | Node identification |
| `cpu.cores` | CPU core count | Default process count |
| `cpu.numa_nodes` | NUMA node count | CPU pinning strategy |
| `memory.total_gb` | Total memory (GB) | Message size limit |
| `gpu.nvidia_present` / `.nvidia_count` | NVIDIA GPU status | Communication arch decision |
| `gpu.nvidia_devices[].name` | GPU model details | Arch selection options |
| `cuda.found` / `.version` | CUDA toolkit | NCCL / wrapper compilation |
| `nccl.found` | NCCL library | NCCL arch availability |
| `mpi.found` / `.implementation` / `.compiler` | MPI implementation | MPI arch + compilation |
| `ib.found` / `.devices` | InfiniBand | UCX / NCCL network config |
| `network.default_interface` | Default network interface | UCCL_SOCKET_IFNAME |
| `ucx.transports` | UCX available transports | UCX_TLS configuration |
| `slurm.partitions` | SLURM partitions | Job configuration |
| `slurm_allocation.*` | Current SLURM allocation | Job config defaults |
| `compiler.gcc` / `.gxx` | Compiler versions | Compilation options |
| `plugin_projects.ccl` | CCL plugin list | Communication library options |
| `plugin_projects.compression` | Compression plugin list | Compression options |
| `disk.ccbench_available_gb` | Available disk space | Dataset sizing |
| `datasets` | Existing datasets | Data source configuration |
| `current_config.*` | Current config values | Migration/modification baseline |

---

## Step 2: Analyze & Display Detection Results

Present a **concise environment summary** to the user in natural language. Example:

```
🔍 Environment Detection Results:

Host: ln01
CPU: 48 cores x AMD EPYC 7402 24-Core (4 NUMA nodes) | Memory: 503 GB
GPU: No local NVIDIA GPU detected | CUDA: Not installed
MPI: Available via mpiexec
InfiniBand: mlx5_0, mlx5_1 (100Gb/s Ethernet)
Network: enp194s0f0np0 (10.252.14.201/24)
UCX: rc_mlx5, dc_mlx5, ud_mlx5, tcp transports available
SLURM: gpu_h200 (2 nodes x 8 GPUs), gpu_h100, gpu_a800
Available CCL plugins: coccl, uccl, zccl
Available compression plugins: dietgpu, SDP4Bit, SZ3, zfp, zstd
Current config: nccl_allreduce (NCCL arch)
Datasets: 20 existing
Disk: ~350TB available
```

> ⚠️ **Detection logic notes**:
> - GPU not detected locally + SLURM partition has GPUs → "Current node has no GPU, but SLURM allocation nodes have GPUs. Config will target SLURM nodes."
> - CUDA not found + NCCL selected → Warning: "CUDA unavailable, NCCL compilation may fail"
> - MPI without compiler (only mpiexec) → Mark as "partial availability"

---

## Step 3: Interactive Q&A

Based on the detection results, ask the user the following configuration questions in order. **Each question provides an intelligent default based on detection results.**

### Question 1: Communication Architecture

Determined by `gpu.nvidia_present` / `cuda.found` / `nccl.found` / `mpi.found`:

```
Currently supported: mpi / nccl / rccl

Which communication architecture would you like to use? [default: <detection-based>]

Rules:
- NVIDIA GPU + NCCL found → default nccl
- MPI compiler found → default mpi
- Only mpiexec → default mpi (but warn that mpicc is needed for benchmark compilation)
- SLURM has GPU partition but no local GPU → default nccl (targeting SLURM nodes)
```

### Question 2: Benchmark Type

```
Available benchmark types:
- mpi_allreduce / nccl_allreduce   [default]
- mpi_sendrecv / nccl_sendrecv
- mpi_bcast / nccl_bcast
- mpi_reducescatter / nccl_reducescatter
- mpi_pingpong / nccl_pingpong
- mpi_overlap / nccl_overlap
- ... (all 12 types, matching the selected arch)

Which one would you like to run? [default: <arch>_allreduce]
For multiple types, use comma separation, or type: benchmark_list to see all
```

### Question 3: Custom Communication Library

Based on `plugin_projects.ccl`:

```
Detected available CCL plugins: <list>

Which communication library would you like to use?
  0) Native <arch> (bare, no wrapper)  [default]
  1) UCCL  ← existing wrapper: nccl_via_uccl.cu
  2) ZCCL  ← existing wrapper: mpi_zccl_wrapper.c
  3) COCCL ← check if wrapper exists (plugin_projects/coccl/src may have one)
  <If a plugin is not detected, gray it out / make it unavailable>

[default: 0] (native is safest when unsure)
```

> If the selected library **has an existing wrapper**, record its path for Step 5.
> If the selected library **does not have an existing wrapper**, notify the user and proceed to Step 5's wrapper generation flow.

### Question 4: Compression

```
Detected available compression plugins: <list>

Enable compression?
  0) Disabled [default]
  1) SZ3
  2) ZFP
  3) DietGPU
  4) SDP4Bit

For multiple selections, use comma separation
```

### Question 5: Job Scheduler

```
Detected SLURM: <partition list>

Current SLURM allocation: <job_id> <nodelist>

Job scheduling method:
  1) slurm (sbatch)    — Submit via sbatch
  2) srun              — Run within existing allocation [default]
  3) local             — Single-node local execution
```

If slurm/srun is selected, ask further:

```
Select partition: [default: first mixed/alloc partition]
Node count: [default: SLURM allocation nodes / 2]
Tasks per node: [default: 8 or CPU cores/8]
Time limit: [default: 01:00:00]
```

### Question 6: Message Sizes

```
Total memory: <total_gb>GB
Suggested message size range: 256MB ~ <total_gb*0.4>MB

Mode:
  1) range — define range + step
  2) list  — explicit size list

Which mode? [default: range]
If range: start <256MB> end <upper_limit> step <256MB>
```

### Question 7: Deviation Metrics

```
Available metrics: <files under deviation_metric_examples/>

Which metrics to enable? [default: mse, mae, cos_sim]
```

### Question 8: Background Daemons

```
Available daemons: cpu, gpu_nvidia, ib, pcie, stress_cpu, stress_gpu_nvidia

Enable daemons? [default: no]
If enabled, select:
  During Phase 1 (performance measurement): <default: cpu, ib>
  During Phase 2 (perf breakdown): <default: cpu, ib>
  CPU load: <30%>
  GPU load: <30%>

Note: gpu_nvidia / stress_gpu_nvidia require CUDA
```

### Question 9: Data Source

```
Existing datasets found: <N>

Data source mode:
  1) file — use existing file [default]
  2) synthetic — generate at runtime
  3) rank_linear — generate linearly by rank

If file, specify path: [default: dataset/weight/weight_4gb.f32]
```

---

## Step 4: Generate Configuration Files

Based on the user's answers + detection results, write each of the 8 JSONC configs one by one.
Use the `Write` or `Edit` tool to modify files under `userconfig/config_in_jsonc/`.

### 4.1 `bench_basic_config.jsonc`

```jsonc
{
  "benchmark_type": "<user choice>",
  "communication_arch": "<user choice>",
  "data_source": {
    "type": "<user choice>",
    "file_path": "<path>"  // if type is "file"
  },
  "message_sizes": {
    "mode": "range",
    "min": 268435456,   // 256MB
    "max": <auto-computed>,
    "add_increment": 268435456  // 256MB
  },
  "iterations": {
    "warmup": 3,
    "measurement": 5,
    "Phase 1": { "warmup": 3, "measurement": 5 },
    "Phase 2": { "warmup": 3, "measurement": 5 }
  },
  "output": {
    "csv_result": "results/<benchmark_type>.csv",
    "binary_output": "results/AllReduce"
  }
}
```

> ⚠️ Preserve the original comment structure of each file; only modify key field values.

### 4.2 `communication_lib_selection.jsonc`

Set `mode` and corresponding `library_paths` / `source_dirs` / `include_dirs` based on the user's library choice.

Existing wrapper template mapping:

| Library | mode | library_paths | source_dirs |
|---|---|---|---|
| Native (bare) | 0 | — | — |
| Native (direct-lib) | 1 | — | — |
| UCCL | 2 | `plugin_projects/ccl/uccl/p2p/libuccl_p2p.so` | `userconfig/wrapper_code_examples/communication/nccl_via_uccl` |
| ZCCL | 2 | `plugin_projects/ccl/zccl/src/.libs/libzccl.so` | `userconfig/wrapper_code_examples/communication/zccl` |

If the selected library is not in this mapping, proceed to Step 5's wrapper generation.

### 4.3 `compression_kernel_selection.jsonc`

Set `mode` and corresponding paths based on the user's compression choice.

| Scheme | mode | Description |
|---|---|---|
| Disabled | 0 | bare |
| SZ3 (CPU) | 2 | `sz3_for_zccl` wrapper |
| ZFP (CPU) | 2 | `zfp_for_zccl` wrapper |
| DietGPU | 3 | Pure code wrapper |
| SDP4Bit | 3 | Pure code wrapper |

### 4.4 `perf_kernel_selection.jsonc`

Select perf wrapper based on communication architecture:

| Arch | source_dirs |
|---|---|
| nccl (w/ wrapper) | `perfcoccl` |
| mpi | `perfmpi` |
| uccl | `perfuccl` |
| zccl | `perfzccl` |

### 4.5 `daemon_config.jsonc`

Write daemon lists and stress parameters based on the user's selections.

### 4.6 `deviation_config.jsonc`

Write the user's selected metric name list.

### 4.7 `environment_variables.jsonc`

Set environment variables based on detection results:

```
# UCX_TLS:
- rc_mlx5 available → "rc,self,sm"
- Only tcp        → "tcp,self,sm"

# UCX_NET_DEVICES:
- mlx5 devices found → "mlx5_0:1"
- No IB            → <default_iface>:1

# UCCL_IB_HCA:
- IB devices found → "mlx5_0,mlx5_1,..."
- No IB          → leave empty

# UCCL_SOCKET_IFNAME:
- Default interface found → <default_interface>
- Not found → "bond0"
```

### 4.8 `job_config.jsonc`

Write based on the user's scheduler choice and SLURM detection results.

---

## Step 5: Wrapper Generation (Optional, Done Last)

> ⚠️ **Why last**: Generating wrapper code requires reading existing templates, understanding API interfaces, and writing new code — all of which consume significant context. Complete all config file generation first, then use the remaining context for wrapper generation.

### 5.1 Check for Existing Wrapper

First, check if code already exists under `userconfig/wrapper_code_examples/<type>/<lib_name>/`.
If it exists, simply update `communication_lib_selection.jsonc` or `compression_kernel_selection.jsonc` to reference it.

### 5.2 Generate a New Wrapper

If the required library **does not have an existing wrapper**, follow these steps:

1. **Read the target library's header files** to understand its API interface
   - For communication libraries: look for `allreduce` / `send` / `recv` function signatures
   - For compression libraries: look for `compress` / `decompress` function signatures
2. **Find the closest existing wrapper to use as a template**
   - Communication wrapper reference: `userconfig/wrapper_code_examples/communication/nccl_via_uccl/`
   - Compression wrapper reference: `userconfig/wrapper_code_examples/compression/sz3/`
3. **Generate the new wrapper file**, write to `userconfig/wrapper_code_examples/<type>/<lib_name>/`
4. **Update `communication_lib_selection.jsonc` or `compression_kernel_selection.jsonc`** to reference the new file

### 5.3 Limitations of Wrapper Generation

- The agent cannot compile or verify the wrapper is correct
- Complex API mapping (e.g., NCCL communication → custom communication library) requires understanding both APIs
- After generation, advise the user: *"Wrapper generated. It is recommended to run `bash scripts/build_script.sh` to verify compilation."*

### 5.4 User Custom Wrappers

If the user wants to write their own wrapper, the framework provides placeholder directories under `userconfig/your_wrapper_code/`:

```
userconfig/your_wrapper_code/
  communication/user_wrapper_mpi/
  communication/user_wrapper_nccl/
  communication/user_wrapper_rccl/
```

Advise the user to place custom wrappers in the appropriate directory, then update the selection JSONC.

---

## Step 6: Summary & Confirmation + Disclaimer

### 6.1 Display Full Configuration Summary

Aggregate all configuration into a **readable config overview table**. Example format:

```
╔══════════════════════════════════════════════════════╗
║              CCBench Configuration Summary           ║
╠══════════════════════════════════════════════════════╣
║ Communication Arch:  nccl  (NCCL AllReduce)          ║
║ Comm Wrapper:        UCCL  (lib+wrapper mode)         ║
║ Compression:         Disabled                         ║
║ Perf Wrapper:        perfcoccl                        ║
║ Message Sizes:       256MB ~ 1280MB (step 256MB)      ║
║ Iterations:          warmup=3, measurement=5          ║
║ Job Scheduler:       srun, partition=gpu_h200, 2n x8 ║
║ Data Source:         file (dataset/weight/...)        ║
║ Deviation Metrics:   mse, mae, cos_sim                ║
║ Daemons:             Disabled                         ║
║ UCX_TLS:             rc,self,sm                       ║
║ UCX_NET_DEVICES:     mlx5_0:1                         ║
║ UCCL_IB_HCA:         mlx5_0,mlx5_1,mlx5_4,mlx5_5     ║
╚══════════════════════════════════════════════════════╝
```

### 6.2 Confirmation

`User, these are all the configuration settings. Please confirm if everything looks correct? [y/N]`

If the user says "no" or "needs changes", return to the corresponding step and modify.

### 6.3 Disclaimer

After the user confirms, print the following notice:

```
⚠️ Disclaimer

This Skill does its best to generate a reasonable configuration based on
environment detection results and your input, but does **NOT guarantee**
that the configuration will work on the first try.

Possible reasons for failure include:

1. Limitations of environment detection (missing dynamic libraries, version incompatibilities, etc.)
2. Wrapper code API adaptation issues
3. Differences in SLURM / MPI cluster environments
4. Installation / build status of third-party libraries (UCCL/ZCCL/compression libs)

If the first run fails, we recommend:
- Run bash scripts/build_script.sh to check for compilation errors
- Check dynamic library dependencies (ldd bin/libs/*.so)
- Have AI analyze the runtime logs or error output to iteratively fix the configuration

The goal of this Skill is to **reduce 80% of the initial configuration effort**,
not to solve all compatibility issues in one shot.
```

---

## Registering This Skill with Claude Code

### Method 1: Reference via CLAUDE.md (Recommended)

Create a `CLAUDE.md` at the project root with the following content:

```markdown
## Available Skills

### /configure-bench
Automatically configure CCBench via environment detection + interactive Q&A.
Usage: type `/configure-bench` in conversation.
Details: see `autosetup_skill/skill.md`.
```

### Method 2: Add to `.claude/settings.json`

Add to `.claude/settings.json`:

```json
{
  "skills": {
    "configure-bench": {
      "description": "Auto-detect environment and configure all CCBench JSONC config files",
      "instructions": "autosetup_skill/skill.md",
      "triggers": ["configure-bench", "setup", "auto-config"]
    }
  }
}
```

### Usage

Invoke the Skill in Claude Code conversation with any of these trigger phrases:

```
/configure-bench
setup CCBench
auto-configure CCBench
/configure
```

---

## Notes

1. **JSONC Comments**: Config files use JSONC format (C-style comments). Use `Write` / `Edit` tools and preserve the comment structure.
2. **Modify Only Key Fields**: Do not rewrite entire files; only change the values that need updating.
3. **Detection Failure Handling**: If a detection command fails, `detect_env.sh` returns empty/default values. The agent should lower the priority of that option accordingly.
4. **Context Management**: If the user needs multiple wrappers (communication + compression), generate and confirm them one at a time to avoid outputting too much code at once.
5. **Permissions**: Modifying files under `userconfig/` does not require special permissions, but running the build scripts may.

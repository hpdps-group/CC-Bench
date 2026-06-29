# CCBench — High-Performance Communication Compression Benchmark Framework

CCBench is a modular benchmark framework for HPC scenarios, supporting **MPI / NCCL / RCCL** communication backends. It leverages `LD_PRELOAD`-based instrumentation to decouple communication compression, performance profiling, and background monitoring.

## Workflow

```
Config (userconfig/*.jsonc) → Build (build_script.sh) → Run (scripts/run/*)
```

1. Configure benchmark parameters in `userconfig/config_in_jsonc/`
2. Run `scripts/build_script.sh` — compiles wrapper libraries and generates run scripts
3. Execute the generated scripts (local / srun / sbatch) to start testing

---

## 🤖 Automatic Configuration via Claude Code

CCBench ships with a **Claude Code Skill** (`configure-bench`) that can automatically detect your environment, ask you for key decisions, and generate all configuration files.

The skill is located at `.claude/skills/configure-bench/` and is **auto-discovered** by Claude Code — no registration needed. Just run Claude Code in the project root and the skill is ready to use.

### Usage

In Claude Code (working in the `CCBench/` directory), just say:

```
/configure-bench
```

Or describe what you want:

```
setup CCBench with NCCL AllReduce + UCCL wrapper
```

The agent will:

1. **Detect** — Probe hardware (CPU/GPU/IB), network, SLURM, compilers, plugins
2. **Analyze** — Present a concise environment summary
3. **Ask** — Guide you through key configuration decisions with smart defaults
4. **Generate** — Write all 8 JSONC config files
5. **Wrapper** — Optionally generate communication/compression wrapper code
6. **Confirm** — Show a full configuration summary for your final approval

> ⚠️ The skill provides intelligent defaults based on your environment, but **does not guarantee first-run success**. If the build or run fails, share the error logs with Claude for iterative fixes. See `.claude/skills/configure-bench/SKILL.md` for the full workflow documentation.

---

## Quick Start

### 1. Environment Setup

Load the required compilers, communication libraries, accelerator SDKs, etc.:

```bash
source scripts/configure_modules.sh   # interactive selection
# or manually load modules
module load compiler/dtk/22.04.2
module load mpi/hpcx/gcc-7.3.1
# ...
```

### 2. Edit Configuration

All configuration files are under `userconfig/config_in_jsonc/`:

| File | Purpose |
|---|---|
| `bench_basic_config.jsonc` | Benchmark type, communication backend, data source, message size, iterations |
| `job_config.jsonc` | Job scheduling method (slurm/srun/local) and resource requests |
| `communication_lib_selection.jsonc` | Communication wrapper selection (mode 0/1/2/3) |
| `compression_kernel_selection.jsonc` | Compression wrapper selection |
| `perf_kernel_selection.jsonc` | Performance profiling wrapper selection |
| `daemon_config.jsonc` | Background daemon selection and configuration |
| `deviation_config.jsonc` | Error/accuracy metric configuration |
| `environment_variables.jsonc` | Runtime environment variable injection |

**Wrapper modes:**

- **mode 0** — Bare-metal run, no wrapper loaded
- **mode 1** — Use precompiled `.so` libraries
- **mode 2** — Precompiled `.so` + source wrapper co-compilation
- **mode 3** — Pure source compilation of wrapper

### 3. Build & Generate Run Scripts

```bash
./scripts/build_script.sh
```

This will:
- Parse all JSONC configuration files
- Build communication / compression / perf wrappers (`.so`) according to config
- Generate run scripts under `scripts/run/` based on `job_choice` in `job_config.jsonc`

### 4. Run

Depending on `job_choice`:

```bash
# local mode (single-node direct execution)
bash scripts/run/run_benchmark.sh

# srun mode (allocate resources via salloc first, then execute)
salloc -N4 -n32 ...
bash scripts/run/run_benchmark.srun.sh

# slurm mode (submit directly)
sbatch scripts/run/run_benchmark.slurm
```

### 5. View Results

- Benchmark data is written to the configured CSV path (default: `results/`)
- Error/accuracy metrics are controlled by `deviation_config.jsonc`
- Profiling data is written to `perf_files/`
- Daemon monitoring output is under `daemon_signals/`

---

## Writing Custom Code

The framework provides three extension points, each with official examples:

### Wrapper (Function Interception / Instrumentation)

Add performance recording or compression logic around your communication functions.

- **Communication wrapper** (`your_wrapper_code/communication/`): Intercept communication functions, interface with custom communication libraries
  - Example: `wrapper_code_examples/communication/zccl/`
- **Compression wrapper** (`your_wrapper_code/compression/`): Intercept compression-related functions, interface with custom compression libraries
  - Example: `wrapper_code_examples/compression/sz3_for_zccl/`
- **Performance wrapper** (`your_wrapper_code/perf/`): Intercept target functions, record performance metrics
  - Example: `wrapper_code_examples/perf/perfmpi/perfmpi.c`

**Available Helper APIs:**

- `perf_helper.h` — General performance recording API: `perf_init()`, `perf_notedown()`, `perf_flush_to_file()`, `perf_get_time()`
- `perf_helper_mpi.h` — MPI TLS state management: `perf_mpi_get_tls()`, `perf_mpi_flush()`, `perf_mpi_msg_size()`
- `nccl_compress_tools.h` — NCCL compression tools: `nccl_send_compressed()`, `nccl_recv_decompress()` (weak symbols, overridable at link time)
- `daemon_helper.h` — Daemon framework: signal file communication, polling loops, etc.

**Wrapper template (perf example):**

```c
#define _GNU_SOURCE
#include <dlfcn.h>
#include "perf_helper.h"
#include "mpi/perf_helper_mpi.h"

// Define the same signature as the target function
int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm) {
    static int (*real)(...) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "MPI_Send");

    double t0 = perf_get_time();
    int ret = real(buf, count, datatype, dest, tag, comm);
    double t1 = perf_get_time();

    perf_notedown(perf_mpi_get_tls(), "MPI_Send", t0, 2,
        (perf_attr_t[]){{"duration", t1 - t0}, {"msg_bytes", ...}});
    return ret;
}
```

After writing your code, configure **mode 2 or 3** in the appropriate JSONC selection file, provide the source path and output name. `build_script.sh` will handle the compilation.

### Daemon (Background Monitoring)

Collect system-level performance data in the background during benchmark runs.

- File naming: `daemon_<name>.c`, must contain a `main()` function
- The framework provides a signal-file communication mechanism
- Examples: `daemon_code_examples/daemon_cpu.c`, `daemon_gpu_nvidia.c`
- Place in `your_daemons/` and run `scripts/register_daemons.sh` to compile

### Deviation Metric (Error Measurement)

Verify data fidelity after compression-decompression cycles.

- File naming: `<name>.c`, must contain a `compute_<name>()` function
- Examples: `deviation_metric_examples/mse.c`, `mae.c`, `psnr.c`
- Place in `your_deviation_metrics/` and run `scripts/register_metrics.sh` to register

---

## Built-in Integrated Libraries

Third-party libraries under `plugin_projects/`, ready to use by path in JSONC configuration:

| Category | Libraries |
|---|---|
| Compression | SZ3, ZFP, DietGPU, SDP4Bit |
| Communication | ZCCL (ZeRO-Cost Communication), UCCL |

---

## Two-Phase Benchmarking

Generated run scripts execute a **two-phase test** by default:

1. **Phase 1 (Baseline)** — Loads only compression/communication wrappers, measures raw performance (multiple iterations, steady-state value)
2. **Phase 2 (Decomposition)** — Additionally loads perf wrapper(s) and starts daemon monitoring for fine-grained performance breakdown

Iteration counts (warmup / measurement) are configured separately in `bench_basic_config.jsonc`.

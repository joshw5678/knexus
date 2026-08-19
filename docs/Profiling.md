# Profiling KNexus Workloads

KNexus can hand off any schedule run to an external profiler. On CUDA, enabling
profiling on a stream wraps every schedule executed on it in an NVTX range and
brackets it with `cudaProfilerStart`/`cudaProfilerStop`, so a tool like NVIDIA
Nsight Systems (`nsys`) captures exactly the annotated region instead of the
whole process. This guide walks through building a workload, running it under
`nsys`, and turning the capture into results you can look at.

Currently implemented for the **CUDA** plugin (`plugins/cuda/cuda_runtime.cpp`).

## 1. Enable profiling in your workload

Profiling is a stream setting. Create the stream with
`NXS_StreamSettings_Profiling`, then run schedules on it as usual:

```cpp
#include <knexus.h>

auto dev0 = runtime.getDevice(0);
auto stream0 = dev0.createStream(NXS_StreamSettings_Profiling);

auto sched = dev0.createSchedule();
auto cmd = sched.createCommand(kernel);
cmd.setArgument(0, buf0);
cmd.finalize({32, 1, 1}, {32, 1, 1}, 0);

sched.run(stream0, NXS_ExecutionSettings_Timing);
```

Every `sched.run(stream0, ...)` call opens an NVTX range named
`nxsRunSchedule(<id>)` and starts/stops CUDA profiler capture around it. Runs
on a stream created *without* `NXS_StreamSettings_Profiling` are invisible to
the profiler — use this to isolate the workload you care about from setup
code (buffer allocation, warmup, etc).

`test/cpp/test_profiling.cpp` is a ready-to-use example: it loads a kernel,
creates a profiling-enabled stream, and runs the kernel in a loop
(`kNumRuns`, default 20). Point it at any kernel file/kernel name/runtime to
profile a different workload, or copy it as a starting point for your own.

## 2. Build

```bash
cd build
cmake --build . --target test_profiling -j$(nproc)
```

(Swap `test_profiling` for your own test binary's CMake target if you wrote a
new one — `test/cpp/CMakeLists.txt` globs `*.cpp` automatically.)

## 3. Set the runtime/device search paths

KNexus discovers plugins and device JSON relative to the current directory by
default, so point it at the build output explicitly. Replace `/path/to/build`
below with the absolute path to your actual `build` directory (e.g. `$(pwd)`
if you're already there) — these are placeholders, not literal paths to copy:

```bash
export KNEXUS_RUNTIME_PATH=/path/to/build/runtime_libs
export KNEXUS_DEVICE_PATH=/path/to/build/device_lib
export LD_LIBRARY_PATH=/path/to/build:/path/to/build/src:/path/to/build/runtime_libs:$LD_LIBRARY_PATH
```

For example, from a `build` directory at
`/home/josh/knexus/knexus-profiler/build`:

```bash
export KNEXUS_RUNTIME_PATH=/home/josh/knexus/knexus-profiler/build/runtime_libs
export KNEXUS_DEVICE_PATH=/home/josh/knexus/knexus-profiler/build/device_lib
export LD_LIBRARY_PATH=/home/josh/knexus/knexus-profiler/build:/home/josh/knexus/knexus-profiler/build/src:/home/josh/knexus/knexus-profiler/build/runtime_libs:$LD_LIBRARY_PATH
```

Note `LD_LIBRARY_PATH` needs `build` and `build/src` themselves (where
`libknexus.so`/`libknexus-api.so` are built), not `build/lib` — that
directory only holds the bundled gtest/gmock libraries.

## 4. Run under `nsys`

```bash
cd build/test/cpp
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  -t cuda,nvtx \
  -o my_workload \
  --force-overwrite=true \
  -- ./test_profiling cuda ../../cuda_kernels/add_vectors.ptx add_vectors
```

Notes on the flags:

- `--capture-range=cudaProfilerApi` — only capture inside the
  `cudaProfilerStart`/`Stop` brackets `NXS_StreamSettings_Profiling` adds.
- `--capture-range-end=repeat` — **use this if your workload calls
  `sched.run()` more than once on the profiling stream.** Each run opens and
  closes the profiler range independently; `repeat` re-arms capture on every
  cycle and writes one `.nsys-rep` per run (`my_workload.1.nsys-rep`,
  `my_workload.2.nsys-rep`, ...). The default (`stop`) only captures the
  *first* range and then stops collecting for the rest of the process — fine
  for a single profiled run, misleading for a loop.
- `-t cuda,nvtx` — trace CUDA kernels/API calls and NVTX ranges. Add `nvml` or
  others as needed.

## 5. Look at the results

**GUI (richest view):** open the `.nsys-rep` file(s) in the Nsight Systems UI
(`nsys-ui my_workload.1.nsys-rep`, or copy the file to a machine that has the
GUI installed) for the full timeline — kernel launches, NVTX ranges, CUDA API
calls, all on one zoomable axis.

**CLI / scripting:** export summary or per-event data as CSV without the GUI:

```bash
nsys stats \
  --report cuda_gpu_trace,cuda_gpu_kern_sum,nvtx_pushpop_trace \
  --format csv --output . --force-export=true \
  my_workload.1.nsys-rep
```

This produces `my_workload.1_cuda_gpu_trace.csv` (every kernel launch, with
start time, duration, grid/block size), `..._cuda_gpu_kern_sum.csv` (per-kernel
totals/averages), and `..._nvtx_pushpop_trace.csv` (the `nxsRunSchedule` NVTX
ranges — start/end/duration). When `--capture-range-end=repeat` produced
multiple `.nsys-rep` files, run this per file and concatenate, keyed by the
numeric suffix in the filename, to get a per-run table across the whole loop.

From there, chart whatever's useful — kernel duration per run, host-side
range duration (dispatch overhead) per run, throughput, etc. — with any
plotting tool (Python/pandas/matplotlib, a notebook, a quick HTML/JS chart).
There's no bundled graphing script in this repo; the CSVs are the hand-off
point.

## Troubleshooting

**"No runtimes found"** — `KNEXUS_RUNTIME_PATH`/`KNEXUS_DEVICE_PATH` aren't
set (or don't point at the build's `runtime_libs`/`device_lib`); see step 3.
Double check you replaced the `/path/to/build` placeholders with your real
build directory — exporting them verbatim is a common mistake and fails the
same way (the loader just finds nothing at that literal path).

**Only one kernel/range shows up in the trace even though the workload loops**
— you used the default `--capture-range-end=stop` (or omitted the flag) with
a workload that calls `sched.run()` multiple times on the profiling stream.
Switch to `--capture-range-end=repeat` (step 4).

**Nothing captured at all** — confirm the stream was created with
`NXS_StreamSettings_Profiling` (not just `NXS_ExecutionSettings_Timing`,
which only affects `NP_ElapsedTime`, not the profiler hooks).

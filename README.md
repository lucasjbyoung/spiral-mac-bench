# spiral-mac-bench

How far does villa's spiral fitter scale on one Apple Silicon machine? This repo
runs the fitter's own headless driver on PHercParis4 at five z-ranges, from 1,000
slices up to the whole written region (13,000 slices). It records time and memory
for each run, plus two comparison runs. One command reproduces the matrix, and a
second builds the table and chart from the logs.

It holds code, specs and numbers only. There is no scan data, no rendered
surfaces and no ink in this repo. Inputs are downloaded separately, under the
data agreement at https://dl.ash2txt.org/LICENSE.txt.

## Results

See [results/results.md](results/results.md) (generated) and
[results/memory_vs_slices.png](results/memory_vs_slices.png).

Headline, on villa #1925 plus one local fix (`e69e015`, below):

| Slices | z | Step (median) | Wall | MPS driver peak | Peak footprint | Completed |
|---|---|---|---|---|---|---|
| 1,000 | 10000-11000 | 0.32 s | 2.7 min | 12.3 GiB | 21.0 GiB | yes |
| 2,000 | 9500-11500 | 0.50 s | 4.5 min | 18.7 GiB | 31.5 GiB | yes |
| 4,000 | 8500-12500 | 0.87 s | 9.3 min | 32.5 GiB | 56.9 GiB | yes |
| 8,000 | 6500-14500 | 1.87 s | 20.8 min | 73.7 GiB | 114.4 GiB | yes |
| 13,000 | 4000-17000 | 14.85 s | 111 min | 155.5 GiB | 204.6 GiB | yes, by paging to swap |

- The same 1,000-slice spec on the CPU: 0.87 s/step, 2.74x slower than MPS.
- **The fix matters.** On the unmodified #1925 head, MPS copies from freed host temporaries arrive corrupted (`non_blocking=True` on MPS reads the source after the call returns). From 4,000 slices up, the fitter then silently drops hundreds to thousands of verified patches and fits corrupted samples. Same-seed step-400 loss with and without the fix:
  - 4k: 881 against 2,552.
  - 8k: 540 against 4,769.
  - 13k: 457 against 2,125.

  See [results/comparison-unfixed-vs-fixed.md](results/comparison-unfixed-vs-fixed.md), `patches/`, and [scripts/repro_mps_nonblocking.py](scripts/repro_mps_nonblocking.py) (a 20-line repro independent of villa).

## Hardware and software

| | |
|---|---|
| Machine | MacBook Pro 16", Apple M5 Max, 18 CPU cores, 40 GPU cores, 128 GB unified memory (Metal 4) |
| OS | macOS 27.2 (build 26B5091g) |
| Python / torch / numpy | 3.14.8 / 2.11.0 / 2.5.2 (villa `spiral-fitting/.venv` from `uv sync`) |
| villa commit tested | `545c4c0bf7b812aae9cf0b1726f8af8b87eda7fd`, the head of open PR [#1925](https://github.com/ScrollPrize/villa/pull/1925) ("spiral-fitting: run on Apple Silicon (MPS) and CPU when CUDA is absent"), plus local commit `e69e0153357be093eb0da7e10230457bee4fec2c` (`patches/0001-*.patch`, MPS-only copy fix). The first pass on the unmodified head is in `results/unfixed-545c4c0b/` |
| Driver | `spiral-fitting/tests/headless_fit_driver.py` from that commit: the documented headless fit, as the CLI drives it, with mesh export skipped |

## What is run

All runs use PHercParis4 and 401 optimiser steps. The iteration count is held
constant across sizes. Every run uses one process, one device and the default
seed (`optimizer_random_seed` from `Config`).

| Run | Device | z-range | Inputs on |
|---|---|---|---|
| `mps_full_z1000` | MPS | 10000-11000 | verified patches, normals, fibres, winding annotations, outer shell |
| `mps_full_z2000` | MPS | 9500-11500 | same |
| `mps_full_z4000` | MPS | 8500-12500 | same |
| `mps_full_z8000` | MPS | 6500-14500 | same |
| `mps_full_z13000` | MPS | 4000-17000 (tutorial default, whole written region) | same |
| `cpu_full_z1000` | CPU | 10000-11000 | same as `mps_full_z1000` |
| `mps_vponly_z1000` | MPS | 10000-11000 | verified patches, winding annotations, outer shell (no normals, no fibres) |

Off in every run:
- tracks (32 GB download; the fitter default is off);
- `fiber_directions` (default off);
- the dense spacing losses and their inputs (`winding_inference`, `gradient_magnitude`).

The spacing terms are disabled with `dense_spacing_mode: grad_mag` and both
spacing weights set to 0, as in the 1,000-slice run first posted on #1925.
z is in fitter voxels: level 2 of volume 20260411134726, 9.6 um.

The fitter scales its per-step sample counts with the z-range, relative to
9,500 slices. A 13,000-slice step therefore does about 13 times the sampling
work of a 1,000-slice step. Per-step times are not comparable across sizes as
"the same work".

## What is measured

| Field | How |
|---|---|
| Step time | a global `torch.optim` post-step hook timestamps every optimiser step (no extra device sync); median and mean of the gaps |
| Training time | first to last step timestamp |
| Load time | process start to first step (input loading, caches, model build) |
| Wall time | `/usr/bin/time -l` real |
| Peak RSS, peak memory footprint | `/usr/bin/time -l` (`maximum resident set size`, `peak memory footprint`) |
| MPS driver peak | `torch.mps.driver_allocated_memory()` sampled every 0.25 s from a background thread |
| Completed / first failure | the run's exit code and the last line of any traceback |
| Loss | the driver's own `total_loss` at step 400 |

`PYTORCH_ENABLE_MPS_FALLBACK` is unset in every run. An op that MPS lacks
therefore fails the run and names itself; it cannot silently run on the CPU.

## Inputs: download only what is needed

From https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/ (about 15.5 GB of the ~90 GB set):

| Path | Size | Used for |
|---|---|---|
| `umbilicus.json`, `abs_winding.json`, `relative_windings.json`, `same_windings.json` | < 10 MB | always |
| `outer_shell/` | small | always |
| `verified_patches/` | 4.9 GB, 89,237 patch dirs | always |
| `lasagna_inputs/las_008_nx.ome.zarr.respool_g4_pair/` | 10.2 GB | normals (pre-packed resident pool; no `pack_resident_pools.py` step needed) |
| `eval_fibers/` | 0.3 GB, 645 JSONs | fibres |

Not needed: `tracks/` (32 GB), `fiber_directions.npz` (2.7 GB), `patch_normals/`
(8.3 GB), `winding_inference/`, `verified_patches_100226/`, and the other
`lasagna_inputs` pools.

`scripts/fetch_inputs.py` mirrors exactly that list (resumable, skips files already present):

```sh
python scripts/fetch_inputs.py ~/spiral_datasets/PHercParis4
```

Then write `spiral-scroll.json` in the dataset root. The published set does not
ship one:

```json
{
  "schema_version": 1,
  "name": "PHercParis4",
  "voxel_size_um": 9.6,
  "spiral_outward_sense": "CW",
  "base_shape_zyx": [18946, 8174, 8174],
  "paths": {"fibers": "eval_fibers"}
}
```

Two of those keys are not optional in practice:
- `paths.fibers`: the server has no `fibers/` directory. The only published fibres are `eval_fibers/`, in level-0 coordinates. The fitter's fibre loader already scales level-0 fibres by 0.25.
- `base_shape_zyx`: 387 of the 645 fibre files declare `coordinate_base_shape_zyx` (`[75784, 32694, 32694]`, level 0). Without a dataset shape to compare against, the loader rejects every one of them (`base_shape_zyx must contain three positive integers`) and fits with the other 258. `[18946, 8174, 8174]` is level 2, the fitter's frame. With it, all 645 load.

Caveat: `eval_fibers` are, by name, the organisers' evaluation fibres. Using them
as a fit input is fine for a time-and-memory benchmark. It means any fibre-term
loss here is not a held-out score.

## Reproduce

Run from a directory that holds this repo as `spiral-mac-bench/`:

```sh
git clone https://github.com/ScrollPrize/villa && cd villa
git fetch origin pull/1925/head && git checkout -b spiral-bench 545c4c0bf7b812aae9cf0b1726f8af8b87eda7fd
git am ../spiral-mac-bench/patches/0001-spiral-fitting-mps-no-nonblocking-pageable-copies.patch   # skip to reproduce the unfixed pass
cd spiral-fitting && uv sync && cd ../..
cp spiral-mac-bench/patches/spiral-scroll.PHercParis4.json ~/spiral_datasets/PHercParis4/spiral-scroll.json

SPIRAL_DIR=$PWD/villa/spiral-fitting \
SPIRAL_DATASET=~/spiral_datasets/PHercParis4 \
PYTHON=$PWD/villa/spiral-fitting/.venv/bin/python \
RUNS_DIR=$PWD/runs \
spiral-mac-bench/run_matrix.sh        # or name runs: spiral-mac-bench/run_matrix.sh mps_full_z1000 cpu_full_z1000

villa/spiral-fitting/.venv/bin/python spiral-mac-bench/scripts/make_table.py runs spiral-mac-bench/results
```

Each run writes `runs/<run>/{run.log, bench.json, driver_result.json, mps_memory_samples.tsv}`.
Checkpoints are deleted after each run (`KEEP_CHECKPOINTS=1` keeps them).
A finished run is skipped on re-invocation (`FORCE=1` redoes it).

Expected runtimes on the machine above (wall, fixed commit): verified patches only at 1k, 2.2 min; CPU at 1k, 7.4 min; MPS at 1k / 2k / 4k / 8k / 13k, 2.7 / 4.5 / 9.3 / 20.8 / 111 min. The whole matrix takes about 2 h 40 min. The 13k run needs about 35 GB of free disk for swap and briefly writes a checkpoint of several GB.

## Limitations

- 401 steps is a scaling measurement, not a converged fit (the tutorial default is 30,000). Losses are not comparable across sizes, because per-step sample counts scale with the z-range.
- One machine and one run per configuration. MPS is not bitwise deterministic: the same 1k spec gives loss 1112.5 and 1112.6 on two runs. Step times vary a few per cent run to run.
- 13,000 slices exceeds physical memory (128 GiB) and completes only by paging. Its time depends on SSD speed and free disk, and the 8 to 14x per-step jump from 8k to 13k is mostly paging.
- Fibres come from `eval_fibers/`, the only published set, so fibre terms are not a held-out score.
- Tracks, dense spacing losses and the fibre-direction field are off throughout.
- The fix (`e69e015`) has not been run on CUDA here. It leaves CUDA behaviour unchanged by construction (it only changes the `non_blocking` flag off CUDA).

## Files

- `run_matrix.sh`: runs the matrix.
- `specs/*.json`: per-run z-range, step count and config overrides.
- `scripts/bench_fit.py`: wraps the villa driver and records timing and memory.
- `scripts/make_table.py`: builds `results/`.
- `scripts/compare_runs.py`: compares two run sets, for example before and after a fix.
- `scripts/fetch_inputs.py`: downloads only the needed inputs.
- `scripts/repro_mps_nonblocking.py`: the MPS copy bug in 20 lines.
- `patches/`: the villa fix and the `spiral-scroll.json` used.

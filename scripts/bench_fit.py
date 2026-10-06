"""Run villa's headless spiral fit driver once and record timing and memory.

    python bench_fit.py <spec.json> <run_dir>

Environment:
  SPIRAL_DIR      villa/spiral-fitting checkout to run (required)
  SPIRAL_DATASET  PHercParis4 spiral-input root (required; fills spec's dataset_path)
  SPIRAL_CACHE    derived-cache directory (default <run_dir>/../cache)
  FIT_SPIRAL_DEVICE  mps | cpu (read by the fitter itself)

Writes into <run_dir>:
  driver_result.json  the driver's own result file (config, losses, checkpoint structure)
  bench.json          timings, MPS memory peaks, step timestamps summary, outcome
The driver's checkpoint lands in <run_dir>/out/ and is deleted afterwards by
run_matrix.sh unless KEEP_CHECKPOINTS=1.

Per-step times come from a global optimiser post-step hook (no extra device
synchronisation, so the fit runs exactly as it would without the harness).
MPS memory is sampled every 0.25 s from a background thread with
torch.mps.driver_allocated_memory() and torch.mps.current_allocated_memory().
"""

import json
import os
import resource
import sys
import threading
import time
import traceback

T_START = time.time()


def main(spec_path, run_dir):
    spiral_dir = os.environ['SPIRAL_DIR']
    with open(spec_path) as f:
        spec = json.load(f)
    spec['dataset_path'] = os.environ['SPIRAL_DATASET']
    spec['cache_path'] = os.environ.get(
        'SPIRAL_CACHE', os.path.join(os.path.dirname(os.path.abspath(run_dir)), 'cache'))
    os.makedirs(run_dir, exist_ok=True)
    resolved_spec = os.path.join(run_dir, 'spec.resolved.json')
    with open(resolved_spec, 'w') as f:
        json.dump(spec, f, indent=2)

    sys.path.insert(0, os.path.join(spiral_dir, 'tests'))
    sys.path.insert(0, spiral_dir)
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    import headless_fit_driver
    import fit_spiral

    device = os.environ.get('FIT_SPIRAL_DEVICE', '')
    use_mps = device in ('', 'mps') and torch.backends.mps.is_available()

    step_times = []
    register_optimizer_step_post_hook(lambda opt, args, kwargs: step_times.append(time.time()))

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            if use_mps:
                samples.append((time.time() - T_START,
                                torch.mps.driver_allocated_memory(),
                                torch.mps.current_allocated_memory()))
            stop.wait(0.25)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    outcome, error = 'completed', None
    t_fit = time.time()
    try:
        headless_fit_driver.run(resolved_spec, os.path.join(run_dir, 'driver_result.json'),
                                os.path.join(run_dir, 'out'))
    except BaseException as exc:  # record the failure, then re-raise
        outcome = 'failed'
        error = ''.join(traceback.format_exception(exc))
    finally:
        stop.set()
        thread.join()
        t_end = time.time()
        gaps = [b - a for a, b in zip(step_times, step_times[1:])]
        gaps_sorted = sorted(gaps)
        bench = {
            'spec': spec,
            'fit_spiral_file': fit_spiral.__file__,
            'device': device or ('mps' if use_mps else 'cpu'),
            'torch': torch.__version__,
            'iterations_requested': spec['iterations'],
            'steps_observed': len(step_times),
            'seconds_process_to_fit_call': t_fit - T_START,
            'seconds_fit_call_to_first_step': (step_times[0] - t_fit) if step_times else None,
            'seconds_training': (step_times[-1] - step_times[0]) if len(step_times) > 1 else None,
            'seconds_after_last_step': (t_end - step_times[-1]) if step_times else None,
            'seconds_total_in_process': t_end - T_START,
            'step_seconds_mean': (sum(gaps) / len(gaps)) if gaps else None,
            'step_seconds_median': gaps_sorted[len(gaps_sorted) // 2] if gaps else None,
            'step_seconds_p90': gaps_sorted[int(len(gaps_sorted) * 0.9)] if gaps else None,
            'mps_driver_allocated_peak_bytes': max((s[1] for s in samples), default=None),
            'mps_current_allocated_peak_bytes': max((s[2] for s in samples), default=None),
            'mps_recommended_max_bytes': torch.mps.recommended_max_memory() if use_mps else None,
            'ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'outcome': outcome,
            'error': error,
        }
        with open(os.path.join(run_dir, 'bench.json'), 'w') as f:
            json.dump(bench, f, indent=2)
        with open(os.path.join(run_dir, 'mps_memory_samples.tsv'), 'w') as f:
            f.write('t_seconds\tdriver_allocated_bytes\tcurrent_allocated_bytes\n')
            for s in samples:
                f.write(f'{s[0]:.2f}\t{s[1]}\t{s[2]}\n')
        print(f'bench: outcome={outcome} steps={len(step_times)} '
              f'step_median={bench["step_seconds_median"]} '
              f'mps_driver_peak_GiB={(bench["mps_driver_allocated_peak_bytes"] or 0) / 2**30:.2f}',
              flush=True)
    if outcome != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    if sys.argv[1:] in (['-h'], ['--help']):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

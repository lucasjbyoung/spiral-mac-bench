"""Build the results table (Markdown + CSV) and the memory chart from run logs.

    python make_table.py <runs_dir> <out_dir>

Reads <runs_dir>/<run>/{run.log,bench.json,driver_result.json}. Writes
<out_dir>/results.md, results.csv and memory_vs_slices.png. Numbers only:
timings, memory, losses and input counts. Nothing from the scan.
"""

import csv
import json
import os
import re
import sys

GIB = 2 ** 30
ORDER = ['mps_vponly_z1000', 'cpu_full_z1000', 'mps_full_z1000', 'mps_full_z2000',
         'mps_full_z4000', 'mps_full_z8000', 'mps_full_z13000']


def grab(pattern, text, cast=str, last=False):
    hits = re.findall(pattern, text)
    if not hits:
        return None
    hit = hits[-1] if last else hits[0]
    return cast(hit.replace(',', '')) if isinstance(hit, str) else hit


def first_error_line(error):
    if not error:
        return ''
    lines = [line.strip() for line in error.strip().splitlines() if line.strip()]
    return lines[-1][:160] if lines else ''


def final_loss(driver_result):
    metrics = (driver_result or {}).get('metrics') or []
    if not metrics:
        return None, None
    values = metrics[-1]['values']
    for key in ('total_loss', 'loss'):
        if key in values:
            return metrics[-1]['iteration'], values[key]
    return metrics[-1]['iteration'], None


def row_for(run_dir, name):
    log_path = os.path.join(run_dir, 'run.log')
    if not os.path.exists(log_path):
        return None
    log = open(log_path, errors='replace').read()
    bench = json.load(open(os.path.join(run_dir, 'bench.json'))) \
        if os.path.exists(os.path.join(run_dir, 'bench.json')) else {}
    driver = json.load(open(os.path.join(run_dir, 'driver_result.json'))) \
        if os.path.exists(os.path.join(run_dir, 'driver_result.json')) else None
    spec = bench.get('spec', {})
    loss_iter, loss = final_loss(driver)
    satisfied = (driver or {}).get('satisfied_aggregates', {})
    row = {
        'run': name,
        'device': bench.get('device'),
        'slices': (spec.get('z_end', 0) - spec.get('z_begin', 0)) or None,
        'z_range': f"{spec.get('z_begin')}-{spec.get('z_end')}" if spec else None,
        'iterations': spec.get('iterations'),
        'steps_observed': bench.get('steps_observed'),
        'verified_patches_loaded': grab(r'loaded and filtered ([\d,]+)/', log, int),
        'fibres_loaded': grab(r'Loaded ([\d,]+) fiber point collections', log, int),
        'fibres_skipped': grab(r'fiber point collections .*; skipped ([\d,]+)', log, int) or 0,
        'normals_pool_gib': grab(r'lasagna normals: resident pool .*\(([\d.]+) GiB\)', log, float),
        'gate_rejected_patches': sum(int(x) for x in re.findall(r'theta consistency gate rejected (\d+) patch', log)),
        'trainable_parameters': grab(r'trainable parameters: ([\d,]+)', log, int),
        'load_s': round(bench['seconds_process_to_fit_call'] + bench['seconds_fit_call_to_first_step'], 1)
        if bench.get('seconds_fit_call_to_first_step') is not None else None,
        'step_s_median': round(bench['step_seconds_median'], 3) if bench.get('step_seconds_median') else None,
        'step_s_mean': round(bench['step_seconds_mean'], 3) if bench.get('step_seconds_mean') else None,
        'training_s': round(bench['seconds_training'], 1) if bench.get('seconds_training') else None,
        'after_training_s': round(bench['seconds_after_last_step'], 1) if bench.get('seconds_after_last_step') else None,
        'wall_s': grab(r'([\d.]+) real', log, float, last=True),
        'peak_rss_gib': round(grab(r'(\d+)\s+maximum resident set size', log, int, last=True) / GIB, 2)
        if grab(r'(\d+)\s+maximum resident set size', log, int, last=True) else None,
        'peak_footprint_gib': round(grab(r'(\d+)\s+peak memory footprint', log, int, last=True) / GIB, 2)
        if grab(r'(\d+)\s+peak memory footprint', log, int, last=True) else None,
        'mps_driver_peak_gib': round(bench['mps_driver_allocated_peak_bytes'] / GIB, 2)
        if bench.get('mps_driver_allocated_peak_bytes') else None,
        'mps_current_peak_gib': round(bench['mps_current_allocated_peak_bytes'] / GIB, 2)
        if bench.get('mps_current_allocated_peak_bytes') else None,
        'completed': bench.get('outcome') == 'completed' and 'EXIT 0' in log,
        'first_failure': first_error_line(bench.get('error')) or (
            '' if 'EXIT 0' in log else grab(r'EXIT (\d+)', log, str, last=True) and 'nonzero exit'),
        'final_loss_iter': loss_iter,
        'final_loss': round(loss, 1) if isinstance(loss, (int, float)) else None,
        'nonfinite_grad_steps': ((driver or {}).get('metrics') or [{}])[-1].get('values', {}).get('nonfinite_grad_steps'),
        'patch_satisfied_fraction': round(satisfied.get('patches', {}).get('overall_fraction') or 0, 3)
        if satisfied else None,
        'pcl_satisfied_fraction': round(satisfied.get('pcls', {}).get('overall_fraction') or 0, 3)
        if satisfied else None,
    }
    return row


COLUMNS = [
    ('run', 'Run'), ('slices', 'Slices'), ('z_range', 'z'), ('device', 'Device'),
    ('verified_patches_loaded', 'Verified patches'), ('gate_rejected_patches', 'Rejected by theta gate'),
    ('fibres_loaded', 'Fibres'),
    ('trainable_parameters', 'Parameters'), ('load_s', 'Load s'),
    ('step_s_median', 'Step s (median)'), ('training_s', 'Training s'), ('wall_s', 'Wall s'),
    ('peak_rss_gib', 'Peak RSS GiB'), ('peak_footprint_gib', 'Peak footprint GiB'),
    ('mps_driver_peak_gib', 'MPS driver peak GiB'), ('completed', 'Completed'),
    ('final_loss', 'Loss @400'), ('patch_satisfied_fraction', 'Patch area satisfied'),
    ('first_failure', 'First failure'),
]


def main(runs_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = [n for n in ORDER if os.path.isdir(os.path.join(runs_dir, n))]
    names += sorted(n for n in os.listdir(runs_dir)
                    if os.path.isdir(os.path.join(runs_dir, n)) and n not in names and n != 'cache')
    rows = [r for r in (row_for(os.path.join(runs_dir, n), n) for n in names) if r]

    with open(os.path.join(out_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value):
        if value is None:
            return '-'
        if isinstance(value, bool):
            return 'yes' if value else 'no'
        if isinstance(value, int) and value >= 10000:
            return f'{value:,}'
        return str(value).replace('|', '/')

    lines = ['| ' + ' | '.join(h for _, h in COLUMNS) + ' |',
             '|' + '|'.join('---' for _ in COLUMNS) + '|']
    for r in rows:
        lines.append('| ' + ' | '.join(fmt(r[k]) for k, _ in COLUMNS) + ' |')
    with open(os.path.join(out_dir, 'results.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mps = [r for r in rows if r['run'].startswith('mps_full') and r['slices']]
    mps.sort(key=lambda r: r['slices'])
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    for key, label, style in (('mps_driver_peak_gib', 'Metal driver allocation, peak', 'o-'),
                              ('peak_footprint_gib', 'process peak memory footprint', 's--'),
                              ('peak_rss_gib', 'process peak RSS', '^:')):
        pts = [(r['slices'], r[key]) for r in mps if r[key] is not None]
        if pts:
            ax.plot(*zip(*pts), style, label=label)
    for r in mps:
        if not r['completed']:
            ax.axvline(r['slices'], color='grey', lw=0.8)
            ax.text(r['slices'], ax.get_ylim()[1] * 0.95, ' did not complete', fontsize=8, va='top')
    physical_gib = 137438953472 / GIB  # sysctl hw.memsize on the test machine: exactly 128 GiB
    ax.axhline(physical_gib, color='#b03030', lw=0.8)
    ax.text(mps[0]['slices'] if mps else 0, physical_gib + 1.5,
            f'physical memory ({physical_gib:.0f} GiB)', fontsize=8, color='#b03030')
    ax.set_xlabel('z-range fitted (slices, 9.6 um each)')
    ax.set_ylabel('GiB')
    ax.set_title('Spiral fit on Apple M5 Max (MPS), PHercParis4, full inputs, 401 steps', fontsize=10)
    ax.set_xticks([r['slices'] for r in mps])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'memory_vs_slices.png'))


if __name__ == '__main__':
    if sys.argv[1:] in (['-h'], ['--help']):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

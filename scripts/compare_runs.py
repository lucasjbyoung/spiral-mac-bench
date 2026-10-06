"""Compare two run directories run by run (e.g. before and after a fix, same seed).

    python compare_runs.py <runs_a> <runs_b> <label_a> <label_b> > comparison.md
"""
import json
import os
import re
import sys

ORDER = ['mps_vponly_z1000', 'cpu_full_z1000', 'mps_full_z1000', 'mps_full_z2000',
         'mps_full_z4000', 'mps_full_z8000', 'mps_full_z13000']


def summary(run_dir):
    if not os.path.exists(os.path.join(run_dir, 'driver_result.json')):
        return None
    d = json.load(open(os.path.join(run_dir, 'driver_result.json')))
    b = json.load(open(os.path.join(run_dir, 'bench.json')))
    log = open(os.path.join(run_dir, 'run.log'), errors='replace').read()
    agg = d['satisfied_aggregates']['patches']
    return {
        'loss': d['metrics'][-1]['values']['total_loss'],
        'patches_fitted': agg['num_entries'],
        'gate_rejected': sum(int(x) for x in re.findall(r'theta consistency gate rejected (\d+) patch', log)),
        'patch_satisfied': agg['overall_fraction'],
        'step_median_s': b['step_seconds_median'],
        'mps_driver_peak_gib': (b['mps_driver_allocated_peak_bytes'] or 0) / 2 ** 30,
    }


def main(a, b, la, lb):
    print(f'| Run | Loss @400 {la} | Loss @400 {lb} | Patches fitted {la} | {lb} '
          f'| Rejected by gate {la} | {lb} | Patch area satisfied {la} | {lb} '
          f'| Step s {la} | {lb} | MPS driver peak GiB {la} | {lb} |')
    print('|' + '---|' * 13)
    for name in ORDER:
        sa, sb = summary(os.path.join(a, name)), summary(os.path.join(b, name))
        if not sa or not sb:
            continue
        print(f"| {name} | {sa['loss']:.1f} | {sb['loss']:.1f} | {sa['patches_fitted']:,} | {sb['patches_fitted']:,} "
              f"| {sa['gate_rejected']:,} | {sb['gate_rejected']:,} | {sa['patch_satisfied']:.3f} | {sb['patch_satisfied']:.3f} "
              f"| {sa['step_median_s']:.3f} | {sb['step_median_s']:.3f} "
              f"| {sa['mps_driver_peak_gib']:.1f} | {sb['mps_driver_peak_gib']:.1f} |")


if __name__ == '__main__':
    if sys.argv[1:] in (['-h'], ['--help']):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])

"""Compare seed-matched A and C runs: with-logger (canonical) vs no-logger."""
import json
from pathlib import Path

def load_run(p):
    snaps, summary = [], None
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get('kind') == 'snapshot':
                snaps.append(obj)
            elif obj.get('kind') == 'summary':
                summary = obj
    return snaps, summary

print("=" * 80)
print("DETERMINISM CHECK: with-logger vs no-logger, same seeds, same code")
print("=" * 80)
print()
print(f"{'file':14s} {'ticks_w':>8s} {'ticks_n':>8s} {'extinct_w':>9s} {'extinct_n':>9s} {'pop_w':>6s} {'pop_n':>6s} {'snaps_w':>8s} {'snaps_n':>8s} {'verdict':>10s}")
print('-' * 100)

verdicts = []
for arm in 'AC':
    for seed in (0, 1, 2):
        with_p = Path('results_15k_behavioral') / f'{arm}_seed{seed}.jsonl'
        no_p   = Path('det_check_no_log')      / f'{arm}_seed{seed}.jsonl'
        if not with_p.exists() or not no_p.exists():
            print(f"{arm}_seed{seed}    MISSING — with: {with_p.exists()}, no: {no_p.exists()}")
            continue
        ws, wsum = load_run(with_p)
        ns, nsum = load_run(no_p)
        if wsum is None or nsum is None:
            print(f"{arm}_seed{seed}    NO SUMMARY in one file")
            continue
        wt = wsum.get('ticks_completed', 0)
        nt = nsum.get('ticks_completed', 0)
        we = wsum.get('extinction')
        ne = nsum.get('extinction')
        wp = ws[-1].get('n_creatures', 0) if ws else 0
        np_ = ns[-1].get('n_creatures', 0) if ns else 0
        # Compare snapshot trajectories tick-by-tick
        traj_match = (len(ws) == len(ns)) and all(
            ws[i].get('tick') == ns[i].get('tick') and
            ws[i].get('n_creatures') == ns[i].get('n_creatures')
            for i in range(min(len(ws), len(ns)))
        )
        if wt == nt and we == ne and wp == np_ and traj_match:
            verdict = "IDENTICAL"
        elif wt == nt and we == ne:
            verdict = "match-summ"
        else:
            verdict = "DIFFER"
        verdicts.append(verdict)
        print(f"{arm}_seed{seed:<8d} {wt:8d} {nt:8d} {str(we):>9s} {str(ne):>9s} {wp:6d} {np_:6d} {len(ws):8d} {len(ns):8d} {verdict:>10s}")

print()
all_id = all(v == "IDENTICAL" for v in verdicts)
any_diff = any(v == "DIFFER" for v in verdicts)
if all_id:
    print("=> LOGGER IS NON-INVASIVE. Behavioral dataset is unbiased. Proceed to report.")
elif any_diff:
    print("=> LOGGER ALTERS SIMULATION. Behavioral data is biased. Fix logger before report.")
else:
    print("=> Summaries match but trajectories differ in detail. Investigate.")

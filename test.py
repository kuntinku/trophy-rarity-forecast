"""Local test runner for the trophyfc entry.

Usage:
  python3 test.py                 # run main.py on the kit's practice cases, print forecasts
  python3 test.py --score FILE    # additionally score outputs against a reference answer
                                  # JSON (makefun/gta6-trophies-answer/v1), if one exists
  python3 test.py --sim           # simulate the launch-night refresh (tracked 638 -> 3868,
                                  # earned frozen) to build pseudo-answers for p00/p01 and
                                  # score our entry and the starter baseline against them

The simulated answers are NOT real answers (the kit ships none); they only check how the
model behaves under the refresh documented in RULES.txt.
"""

import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = Path("/root/makefun/gta6-trophy-rarity-forecast-01-kit")
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def run_entry(input_dir: Path, out_dir: Path) -> None:
    t = time.perf_counter()
    subprocess.run(
        [sys.executable, str(HERE / "main.py")],
        env={"MAKEFUN_INPUT_DIR": str(input_dir), "MAKEFUN_OUTPUT_DIR": str(out_dir),
             "PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        check=True,
    )
    ms = (time.perf_counter() - t) * 1000
    print(f"run time: {ms:.0f} ms")


def validate(out_path: Path, case: dict) -> list[str]:
    errs = []
    raw = out_path.read_bytes()
    if len(raw) > 65536:
        errs.append("too large")
    obj = json.loads(raw)
    if set(obj) != {"trophies"}:
        errs.append("top-level keys")
    ids = {t["id"] for t in case["trophies"]}
    got = [r["id"] for r in obj["trophies"]]
    if sorted(got) != sorted(ids) or len(set(got)) != len(got):
        errs.append("id set mismatch")
    for r in obj["trophies"]:
        if set(r) != {"id", "q"} or len(r["q"]) != 5:
            errs.append(f"row {r.get('id')} shape")
        for q in r["q"]:
            if not (isinstance(q, (int, float)) and 0 <= q <= 100 and math.isfinite(q)):
                errs.append(f"row {r.get('id')} q value {q}")
        if any(r["q"][k] < r["q"][k - 1] for k in range(1, 5)):
            errs.append(f"row {r['id']} not non-decreasing")
    return errs


def pinball_loss(forecast: dict, answer: dict) -> dict:
    """Mean pinball loss on the logit scale, plus typical miss and 90% coverage."""
    amap = {a["id"]: a["actual"] for a in answer["actuals"]} if "actuals" in answer \
        else {t["id"]: t["actual"] for t in answer["trophies"]}
    fmap = {r["id"]: r["q"] for r in forecast["trophies"]}
    losses, misses, covered, n = [], [], 0, 0
    for tid, y in amap.items():
        qs = fmap[tid]
        ly = logit(y)
        for tau, q in zip(LEVELS, qs):
            d = ly - logit(q)
            losses.append(2 * max(tau * d, (tau - 1) * d))
        misses.append(abs(y - qs[2]))
        if qs[0] <= y <= qs[4]:
            covered += 1
        n += 1
    return {
        "loss": sum(losses) / len(losses),
        "typical_miss": sum(misses) / n,
        "coverage": covered / n,
        "items": len(losses),
    }


def baseline_forecast(case: dict) -> dict:
    """The house persistence baseline, for comparison."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    rows = []
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i] or 0.0
        lg = logit(r)
        qs = []
        for z in Z:
            v = 100 / (1 + math.exp(-(lg + z * spread)))
            v = min(max(v, 0.01), 99.99)
            v = math.floor(v * 10000 + 0.5) / 10000
            qs.append(int(v) if float(v).is_integer() else v)
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def simulate_refresh(case: dict, f: float = 3868 / 638) -> dict:
    """Pseudo-answer: refresh at the target multiplies tracked players, earned frozen."""
    last = case["history"][-1]
    rows = []
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i] or 0.0
        new = r / f if r < 99.995 else 100.0
        rows.append({"id": t["id"], "actual": round(new, 2)})
    return {"trophies": rows}


def main() -> None:
    out_dir = Path(tempfile.mkdtemp(prefix="trophyfc_out_"))
    run_entry(KIT / "input", out_dir)

    ref = None
    for cand in (KIT / "reference" / "answer.json", KIT / "input" / "answer.json"):
        if cand.exists():
            ref = cand
            break

    cases = {}
    for p in sorted((KIT / "input").glob("*.in")):
        case = json.loads(p.read_text())
        cases[p.stem] = case
        errs = validate(out_dir / f"{p.stem}.out", case)
        fc = json.loads((out_dir / f"{p.stem}.out").read_text())
        row0 = fc["trophies"][0]
        print(f"{p.stem}: rows={len(fc['trophies'])} errors={errs or 'none'} "
              f"trophy1 q={row0['q']}")

    if ref:
        answer = json.loads(ref.read_text())
        for stem, case in cases.items():
            fc = json.loads((out_dir / f"{stem}.out").read_text())
            sc = pinball_loss(fc, answer)
            print(f"{stem} vs reference: loss={sc['loss']:.6f} "
                  f"typ_miss={sc['typical_miss']:.3f} cov90={sc['coverage']:.2f}")
        return

    print("\nNo reference answer.json in the kit (expected: targets not yet reached).")
    print("Simulated refresh check (pseudo-answers, NOT real):")
    for stem, case in cases.items():
        answer = simulate_refresh(case)
        fc = json.loads((out_dir / f"{stem}.out").read_text())
        ours = pinball_loss(fc, answer)
        base = pinball_loss(baseline_forecast(case), answer)
        print(f"{stem} sim: ours loss={ours['loss']:.4f} miss={ours['typical_miss']:.3f} "
              f"cov={ours['coverage']:.2f} | baseline loss={base['loss']:.4f} "
              f"miss={base['typical_miss']:.3f} cov={base['coverage']:.2f}")


if __name__ == "__main__":
    main()

"""gta6-trophy-rarity-forecast-01 entry: refresh-aware dilution forecaster (Python 3.13 stdlib only).

Mechanism (from launch-night evidence in the rules): Exophase freezes a game's earned
counts between refreshes; a refresh that multiplies players_tracked by f drops every
rate by roughly f^(-e), where e is the trophy's elasticity of rate to tracked players
(earned grows a little with the influx, so 0 < e < 1; the 100% trophy has e = 0).

Model, per trophy:
  - elasticity e_i: OLS of logit(rate) on ln(players_tracked) over the case's history,
    shrunk toward the pooled elasticity across trophies (robustness on noisy series).
  - expected tracked growth to the target: OLS of ln(players_tracked) on ln(1 + hours)
    (sub-linear adoption growth), projected forward and capped.
  - median logit at target = adj_last + drift*dx + e_i * ln(f), where adj = logit - e*ln(pt)
    and drift is the OLS slope of adj over the recent window.
  - launch-week stale-page prior: tracked flat >= 2 h on a page that should be growing
    means a big refresh is overdue -> fall back to a prior dilution shift with wider spread.
  - spread = sqrt(walk^2 + drift-residual^2 + shift-uncertainty^2). 100.00 rates stay 100.

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output). Deterministic; ~50 ms per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

WALK = 0.20            # residual logit walk per sqrt-day
LNF_CAP = 3.0          # cap on projected ln growth of tracked players
LNF_DAMP = 0.6         # damp the ln(1+t) extrapolation: adoption growth decelerates
STALE_HOURS = 2.0      # tracked flat this long in launch week -> refresh overdue
YOUNG_DAYS = 7.0
PRIOR_LNF = 1.8        # ~6x, the documented launch-night refresh
PRIOR_BETA = 0.8       # dilution when we cannot observe elasticity
UNC_BASE = 0.35        # spread from shift uncertainty
UNC_PER = 0.20


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def round4(x: float) -> float | int:
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_iso(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and residual RMS."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    resid = math.sqrt(sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(xs, ys)) / n)
    return slope, resid


def variance(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / n


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name}: unexpected levels {case['levels']}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name}: empty history")
    last = history[-1]
    d_days = case["horizon_minutes"] / 1440.0
    hours_ahead = d_days * 24.0

    ts = []  # hours since the first read
    t0 = parse_iso(history[0]["at"])
    for h in history:
        ts.append((parse_iso(h["at"]) - t0).total_seconds() / 3600.0)

    pts = [max(1, h["players_tracked"]) for h in history]
    ln_pts = [math.log(p) for p in pts]

    # Expected ln growth of tracked players from the last read to the target:
    # sub-linear fit of ln(pt) on ln(1 + hours), projected and capped.
    xg = [math.log(1.0 + t) for t in ts]
    g_slope, g_resid = ols(xg, ln_pts)
    hours_span = ts[-1] - ts[0]
    if hours_span < 1.0 or g_slope <= 0.0:
        ln_f_obs = 0.0
    else:
        x_last = xg[-1]
        x_target = math.log(1.0 + ts[-1] + hours_ahead)
        ln_f_obs = min(max(0.0, g_slope * (x_target - x_last) * LNF_DAMP), LNF_CAP)

    # Stale-page prior: flat tracked for >= STALE_HOURS on a young game means the page
    # has not refreshed yet and a big jump is overdue around/before the target.
    stale_h = 0.0
    for i in range(len(history) - 1, 0, -1):
        if pts[i] == pts[i - 1]:
            stale_h += ts[i] - ts[i - 1]
        else:
            break
    young = True
    try:
        young = (parse_iso(case["at"]) - parse_iso(case["game"]["release_date"])
                 ).total_seconds() / 86400.0 < YOUNG_DAYS
    except Exception:
        young = True

    use_prior = young and stale_h >= STALE_HOURS and ln_f_obs < 0.3
    ln_f = PRIOR_LNF if use_prior else ln_f_obs

    # Elasticity of logit(rate) to ln(players_tracked), pooled across trophies.
    e_pooled_list = []
    for i in range(len(case["trophies"])):
        ys = [logit(h["rates"][i]) for j, h in enumerate(history)
              if h["rates"][i] is not None and history[j]["rates"][i] < 99.995]
        xs = [ln_pts[j] for j, h in enumerate(history)
              if h["rates"][i] is not None and history[j]["rates"][i] < 99.995]
        if len(xs) >= 4 and variance(xs) > 0.05:
            slope, _ = ols(xs, ys)
            e_pooled_list.append(slope)
    e_pooled = sorted(e_pooled_list)[len(e_pooled_list) // 2] if e_pooled_list else -PRIOR_BETA

    s_walk = WALK * math.sqrt(max(d_days, 0.02))

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate_last = last["rates"][i]
        if rate_last is None:
            rate_last = 0.0
        lg = logit(rate_last)

        if rate_last >= 99.995:
            # Printed 100.00: everyone tracked has it; refreshes keep it at 100.
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue

        # Per-trophy elasticity, shrunk toward the pooled value.
        ys = [logit(h["rates"][i]) for j, h in enumerate(history)
              if h["rates"][i] is not None and history[j]["rates"][i] < 99.995]
        xs = [ln_pts[j] for j, h in enumerate(history)
              if h["rates"][i] is not None and history[j]["rates"][i] < 99.995]
        e = e_pooled
        if len(xs) >= 4 and variance(xs) > 0.05:
            slope_i, _ = ols(xs, ys)
            w = min(1.0, (len(xs) - 2) / 8.0)
            e = w * slope_i + (1.0 - w) * e_pooled
        e = min(max(e, -1.5), 0.0)

        # Earn-adjusted logit (rate with the dilution explained away) and its drift.
        adj = [y - e * x for x, y in zip(xs, ys)]
        a_ts = [ts[j] for j, h in enumerate(history)
                if h["rates"][i] is not None and history[j]["rates"][i] < 99.995]
        slope_a, resid_a = ols(a_ts, adj)
        if len(a_ts) < 3 or (a_ts[-1] - a_ts[0]) < 1.0:
            slope_a, resid_a = 0.0, 0.0
        slope_a = min(max(slope_a, -0.3), 0.3)
        resid_a = min(resid_a, 1.0)

        shift = e * ln_f + slope_a * hours_ahead
        if use_prior and e == e_pooled and not e_pooled_list:
            shift = -PRIOR_BETA * ln_f  # no observable elasticity: launch-refresh prior
        shift = min(max(shift, -4.0), 1.0)

        median_lg = min(max(lg + shift, logit(0.01)), logit(99.99))

        s_shift = UNC_BASE + UNC_PER * abs(shift)
        s = math.sqrt(s_walk**2 + resid_a**2 + s_shift**2)
        s = max(s, 0.08)

        qs = []
        for z in Z:
            q = 100.0 / (1.0 + math.exp(-(median_lg + z * s)))
            qs.append(round4(min(max(q, 0.01), 99.99)))
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"[trf] {path.name}: {len(forecast['trophies'])} trophies -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

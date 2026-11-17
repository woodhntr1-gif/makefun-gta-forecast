"""GTA VI launch-day Twitch CCU forecaster -- make.fun bounty
gta6-twitch-ccu-forecast-01 (program/v1, Python 3.13, stdlib only).

Exact harness contract (from the competition's public starter / RULES.txt,
GET https://make.fun/api/competitions/gta6-twitch-ccu-forecast-01 and the
public kit's starter/main.py):

  * Run as ``python3 -P -s /bot/main.py`` (PYTHONHASHSEED=0), no network,
    no installs, deterministic, <= 8 s / 1 CPU / 1024 MB per case.
  * Read every ``<id>.in`` in the input directory (``/input`` in the
    grader; ``MAKEFUN_INPUT_DIR`` outside it). One case at a time.
    Case JSON schema ``makefun/twitch-ccu-case/v1``:
    {"schema", "id", "game", "day_start", "checkpoint", "levels",
     "known", "forecast", "history", "history_hourly", "streams",
     "categories_hourly", "comparables", "notes"}
    ``known`` holds hour 1's print {hour: 1, from, until, viewers,
    channels, reads}; ``forecast`` lists the hours to forecast (2..24);
    ``levels`` is [0.05, 0.25, 0.5, 0.75, 0.95]; ``comparables`` holds
    past launches, each with ``day_start`` and ``hourly`` rows covering
    the launch day (viewers per hour, from day_start).
  * Write ``<id>.out`` to the output directory (``/output`` in the
    grader; ``MAKEFUN_OUTPUT_DIR`` outside it):
    {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}
    exactly 23 rows (hours 2..24, each once, no other key), each q five
    finite numbers, > 0 and <= 1e9, non-decreasing; <= 65,536 bytes.
  * Scoring: mean pinball loss on the log scale over scored hours and
    levels, lowest wins; must beat the house persistence baseline
    (every hour = hour-1 print, spread 0.15*sqrt(h-1) in log space).

Flexible fallback (for local / synthetic runs only, documented in
README.md): if no ``*.in`` files are found in the input directory, the
program reads a JSON case -- or a simple {"hour1": N} / plain number /
list of hourly viewers -- from a file path in argv[1] or from stdin,
and writes the forecast JSON to stdout (and to the output dir if set).

Model
-----
1. Empirical (when the case carries ``comparables``): for every
   comparable launch, log-ratios log(viewers_h / viewers_1) over its
   launch day, weighted by a Gaussian kernel on log(hour-1 viewers)
   similarity to this case (bandwidth 1.0 in log space, floor weight
   so every comparable still counts a little). Weighted quantiles of
   the log-ratios per hour give the forecast shape. This re-fits on
   every case from that case's own comparables.
2. Parametric fallback / blend: a launch-day curve anchored on hour 1:
   an early dip from the hour-1 launch spike (power/exponential-style
   decay to a floor of ~0.75x in hours 3-10), recovery as the US
   unlocks (05:00-08:00 UTC) and Europe returns, and a prime-time
   secondary bump peaking around hours 19-21 UTC (~1.45x), easing by
   hour 24. The table below is smoothed from the practice comparables'
   median ratios, so the fallback follows the same shape.
3. Quantiles: lognormal-style. Median = blended (75% empirical /
   25% parametric when the effective sample size is large enough).
   Spread sigma_h grows with horizon (at least ~0.15*sqrt(h-1),
   estimated from the comparables' inter-quantile range when
   available, clipped to [0.18, 1.25]). Quantiles are enforced
   strictly non-decreasing, given a minimum log spacing, and clipped
   to [1, 1e9].

Deterministic: no randomness, no wall clock, no network, stdlib only.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put this file's dir back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
MAX_Q = 1_000_000_000.0

# Parametric median ratio to hour-1 viewers, hours 2..24 (see docstring).
PARAM_RATIO = {
    2: 0.92, 3: 0.82, 4: 0.76, 5: 0.75, 6: 0.76, 7: 0.76, 8: 0.77,
    9: 0.76, 10: 0.78, 11: 0.83, 12: 0.90, 13: 0.97, 14: 1.05,
    15: 1.13, 16: 1.20, 17: 1.27, 18: 1.34, 19: 1.42, 20: 1.45,
    21: 1.44, 22: 1.40, 23: 1.32, 24: 1.18,
}
DEFAULT_ANCHOR = 100_000.0  # only if no viewer data exists anywhere


def round2(x: float):
    """Round to 2 decimals, half up (as the starter / JS Math.round)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def clamp_q(x: float) -> float:
    return min(MAX_Q, max(1.0, x))


# ---------------------------------------------------------------- parsing

def _viewers_of(row):
    if isinstance(row, dict):
        v = row.get("viewers")
        if v is None:
            v = row.get("value", row.get("count"))
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    if isinstance(row, (int, float)):
        return float(row)
    return None


def extract_p1(case) -> float:
    """Hour-1 viewers from any plausible case shape; missing -> 0."""
    if isinstance(case, (int, float)):
        return float(case)
    if isinstance(case, list):
        for row in case:
            v = _viewers_of(row)
            if v is not None:
                return v
        return 0.0
    if not isinstance(case, dict):
        return 0.0
    known = case.get("known")
    if isinstance(known, list):
        for row in known:
            if isinstance(row, dict) and row.get("hour") == 1:
                v = _viewers_of(row)
                if v is not None:
                    return v
        for row in known:  # first known row, whatever its hour label
            v = _viewers_of(row)
            if v is not None:
                return v
    elif known is not None:
        v = _viewers_of(known)
        if v is not None:
            return v
    for key in ("hour1", "hour_1", "p1", "first_hour", "hour1_viewers",
                "viewers"):
        v = case.get(key)
        if isinstance(v, (int, float)):
            return float(v)
        v2 = _viewers_of(v)
        if v2 is not None:
            return v2
    hours = case.get("hours")
    if isinstance(hours, list) and hours:
        v = _viewers_of(hours[0])
        if v is not None:
            return v
    return 0.0


def fallback_anchor(case) -> float:
    """Largest viewer count visible in history fields (for p1 == 0)."""
    best = 0.0
    if isinstance(case, dict):
        for key in ("history", "history_hourly"):
            rows = case.get(key)
            if isinstance(rows, list):
                for row in rows:
                    v = _viewers_of(row)
                    if v is not None and v > best:
                        best = v
    return best if best > 0 else DEFAULT_ANCHOR


# ------------------------------------------------------- empirical model

def comparable_curves(case):
    """[(log_h1, [viewers h1..h24])] from case['comparables']."""
    curves = []
    if not isinstance(case, dict):
        return curves
    comps = case.get("comparables")
    if not isinstance(comps, list):
        return curves
    for comp in comps:
        if not isinstance(comp, dict):
            continue
        hourly = comp.get("hourly")
        day_start = comp.get("day_start")
        if not isinstance(hourly, list) or not day_start:
            continue
        rows = [r for r in hourly
                if isinstance(r, dict) and isinstance(r.get("from"), str)
                and r["from"] >= day_start]
        rows.sort(key=lambda r: r["from"])
        rows = rows[:24]
        if len(rows) < 24:
            continue
        vals = [_viewers_of(r) for r in rows]
        if any(v is None for v in vals) or vals[0] is None or vals[0] <= 0:
            continue
        curves.append((math.log(vals[0]), vals))
    return curves


def weighted_quantile(sorted_pairs, tau):
    """sorted_pairs: [(value, weight)] sorted by value."""
    total = sum(w for _, w in sorted_pairs)
    if total <= 0:
        return sorted_pairs[len(sorted_pairs) // 2][0]
    target = tau * total
    acc = 0.0
    for value, w in sorted_pairs:
        acc += w
        if acc >= target:
            return value
    return sorted_pairs[-1][0]


def empirical_quantiles(case, p1):
    """{hour: [log-ratio quantiles]} and effective sample size."""
    curves = comparable_curves(case)
    if not curves or p1 <= 0:
        return {}, 0.0
    log_p1 = math.log(p1)
    weighted = []
    for log_h1, vals in curves:
        d = (log_p1 - log_h1) / 1.0  # kernel bandwidth 1.0 (log space)
        w = math.exp(-0.5 * d * d) + 0.02  # floor: everyone counts a bit
        weighted.append((w, vals))
    ess = (sum(w for w, _ in weighted) ** 2) / sum(w * w for w, _ in weighted)
    out = {}
    for h in range(2, 25):
        pairs = []
        for w, vals in weighted:
            vh = vals[h - 1]
            if vh is None:
                continue
            lr = math.log((vh + 1.0) / (vals[0] + 1.0))
            pairs.append((lr, w))
        if len(pairs) >= 4:
            pairs.sort(key=lambda t: t[0])
            out[h] = [weighted_quantile(pairs, tau) for tau in LEVELS]
    return out, ess


# ----------------------------------------------------------------- model

def forecast(case) -> dict:
    p1_raw = extract_p1(case)
    p1 = p1_raw if p1_raw > 0 else fallback_anchor(case)
    emp, ess = empirical_quantiles(case, p1)
    use_emp = ess >= 4.0 and len(emp) >= 20
    w_emp = 0.75 if use_emp else 0.0

    # Hours to forecast: the case's own list when present, else 2..24.
    hours_wanted = list(range(2, 25))
    if isinstance(case, dict) and isinstance(case.get("forecast"), list):
        hs = [r.get("hour") for r in case["forecast"]
              if isinstance(r, dict) and isinstance(r.get("hour"), int)]
        if hs:
            hours_wanted = sorted(set(hs))

    rows = []
    for h in hours_wanted:
        hh = min(24, max(2, h))
        log_ratio_param = math.log(PARAM_RATIO[hh])
        # Parametric spread: baseline-like random walk in log space.
        sigma_param = 0.15 * math.sqrt(hh - 1) + 0.10
        if use_emp and hh in emp:
            eq = emp[hh]  # log-ratio quantiles from comparables
            log_med = w_emp * eq[2] + (1 - w_emp) * log_ratio_param
            # Spread from the empirical inter-quantile ranges.
            sig_iqr = (eq[3] - eq[1]) / (2 * 0.6744897501960817)
            sig_90 = (eq[4] - eq[0]) / (2 * 1.6448536269514722)
            sigma_emp = max(sig_iqr, sig_90)
            # Floor at the baseline's random-walk spread, 0.15*sqrt(h-1),
            # as the module docstring states: however tightly the
            # comparables agree, a launch day can wander at least that
            # far by hour h, and a narrower band would under-cover the
            # late hours (the old sigma_param * 0.8 floor fell below
            # 0.15*sqrt(h-1) from about hour 10 on).
            sigma = min(1.25, max(0.6 * sigma_emp + 0.4 * sigma_param,
                                  0.15 * math.sqrt(hh - 1), 0.18))
            log_q = [log_med + z * sigma for z in Z]
            # Blend each non-median quantile 50/50 with the empirical one
            # (shifted onto the blended median) for honest asymmetry.
            for i in (0, 1, 3, 4):
                emp_shifted = log_med + (eq[i] - eq[2])
                log_q[i] = 0.5 * log_q[i] + 0.5 * emp_shifted
        else:
            log_med = log_ratio_param
            sigma = min(1.25, max(sigma_param, 0.18))
            log_q = [log_med + z * sigma for z in Z]
        # Enforce non-decreasing with a minimum log spacing.
        for i in range(1, 5):
            if log_q[i] < log_q[i - 1] + 0.02:
                log_q[i] = log_q[i - 1] + 0.02
        qs = [round2(clamp_q(p1 * math.exp(x))) for x in log_q]
        qs = sorted(qs)  # round2/clipping must never break the ordering
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


# ------------------------------------------------------------------- io

def process_input_dir(in_dir: Path, out_dir: Path) -> int:
    files = sorted(in_dir.glob("*.in")) if in_dir.is_dir() else []
    if not files:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        case = json.loads(f.read_text(encoding="utf-8"))
        result = forecast(case)
        case_id = case.get("id") if isinstance(case, dict) else None
        out_name = f"{case_id}.out" if case_id else f"{f.stem}.out"
        (out_dir / out_name).write_text(
            json.dumps(result, separators=(",", ":")), encoding="utf-8")
    return len(files)


def main() -> None:
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    if process_input_dir(in_dir, out_dir):
        return
    # Flexible fallback: argv file, else stdin (if piped), else empty case.
    case = {}
    if len(sys.argv) > 1 and sys.argv[1] not in ("-",):
        case = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            try:
                case = json.loads(text)
            except json.JSONDecodeError:
                try:
                    case = float(text)
                except ValueError:
                    case = {}
    result = forecast(case)
    sys.stdout.write(json.dumps(result) + "\n")
    # Also honour /output/<id>.out when an output dir was explicitly set.
    if "MAKEFUN_OUTPUT_DIR" in os.environ:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            cid = case.get("id", "case") if isinstance(case, dict) else "case"
            (out_dir / f"{cid}.out").write_text(
                json.dumps(result, separators=(",", ":")), encoding="utf-8")
        except OSError:
            pass


if __name__ == "__main__":
    main()

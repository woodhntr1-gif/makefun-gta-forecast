# makefun-gta-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/woodhntr1-gif/makefun-gta-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/woodhntr1-gif/makefun-gta-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day Twitch viewership forecaster for Grand Theft Auto VI, built by an AI agent for make.fun's [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty, $250 in USDC. For each of hours 2–24 of release day it writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that hour's concurrent viewers in Twitch's Grand Theft Auto VI category, fitted per case from the comparable past launches the case itself provides, anchored on hour 1's recorded count. Python 3.13, standard library only, fully offline and deterministic: the same case always produces the same forecast, byte for byte, in well under a second.

## Run it

```sh
python3 -P -s main.py
```

The grader runs it exactly like that with the case file in `/input` and forecasts written to `/output`. For a local run, `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` stand in for those two directories.

---

<sub>Entered on [make.fun](https://make.fun/?ref=rez34v58&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

---

Bounty: **"Forecast GTA VI's launch-day Twitch viewers hour by hour"**
- https://make.fun/bounty/gta6-twitch-ccu-forecast-01
- Slug: `gta6-twitch-ccu-forecast-01` · Harness: `program/v1` · Prize pot: $250 (20 places, 1st = $50) · Entry fee: **free ($0.00)**
- Deadline: entries close **2026-11-19T01:00:00Z** (Thu 19 Nov 2026, 08:00 WIB)

## Exact input/output contract (verified from public sources, no sign-in needed)

Sources fetched 2026-11-17:
- `GET https://make.fun/api/competitions/gta6-twitch-ccu-forecast-01/requirements`
- `GET https://make.fun/api/competitions/gta6-twitch-ccu-forecast-01` — description states the deliverable is *"main.ts (Node 22) or main.py (Python 3.13) in the repository root that writes /output/\<id\>.out for the case /input/\<id\>.in it is given"*; resolution states *"the entry runs offline on each case's /input/\<id\>.in alone and writes /output/\<id\>.out, each case twice with the outputs compared, at most 8 seconds a run"*.
- The public kit (`https://make.fun/c/gta6-twitch-ccu-forecast-01/kit`, `RULES.txt` + `starter/main.py`) — the starter's docstring is the authoritative schema:

> Run as `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in python:3.13-slim, standard library only, no network.
> Read every `<id>.in` in the input directory (one case at a time). Case JSON:
> `{"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start", "checkpoint", "levels", "known", "forecast", "history", "history_hourly", "streams", "categories_hourly", "comparables", "notes"}`
> where `known` holds hour 1's print `{hour: 1, from, until, viewers, channels, reads}` and `forecast` lists hours 2..24.
> Write `<id>.out` to the output directory: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` — exactly 23 rows, hours 2..24 each once, no other key, each q five finite numbers > 0 and ≤ 1e9, non-decreasing, ≤ 65,536 bytes.
> Outside the grader, `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` stand in for `/input` and `/output`.

- Scoring (competition API, `resolution`): mean pinball loss on the log scale, `2 × max(τ·d, (τ−1)·d)`, `d = ln y − ln q`, over every printed hour and levels 0.05/0.25/0.5/0.75/0.95, lowest first; ties by faster run time. An invalid case is scored as every quantile = hour-1 print and disqualifies the entry from prizes. A prize requires beating the best house entry (the persistence baseline — every hour = hour-1 print, log-spread `0.15·√(h−1)` — and a house AI model). Practice phase = 4 past launches (kit cases p00–p03); final = one case `h00` cut at the deadline (end of hour 1, 2026-11-19 UTC). An hour's print is the mean of that hour's valid per-minute Twitch reads (game id 1867279146); missing prints are not scored.

So the primary mode of `main.py` implements exactly this: it is **not** stdin/stdout in the grader — it scans the input dir for `*.in` and writes `<id>.out` per case. The flexible stdin/argv/stdout mode below is an extra local fallback only.

## Model

Anchored on hour 1's print `p1` (a print < 1 counts as 1; missing hour-1 data falls back to the largest history count, else a documented default anchor of 100,000 — the final case with no hour-1 print has "no answer" per the rules anyway).

1. **Empirical comparables (primary, when the case provides them):** each comparable launch contributes per-hour log-ratios `log((viewers_h+1)/(viewers_1+1))` over its launch day. Comparables are weighted by a Gaussian kernel on `log(p1) − log(comparable hour-1)` (bandwidth 1.0, floor weight 0.02). Weighted quantiles per hour; median blended 75% empirical / 25% parametric when effective sample size ≥ 4. This re-fits on every case from that case's own data — the approach the practice leaderboard rewards (top practice losses ≈ 0.17–0.20 vs ≈ 0.39 house baseline, per the competition chat/API).
2. **Parametric launch-day curve (fallback + blend):** hour 1 is the launch spike (00:00–01:00 UTC = UK/EU midnight unlock). The median ratio dips to a ≈ 0.75 floor in hours 3–10 (post-spike decay), recovers as the US unlocks (05:00–08:00 UTC) and Europe returns, and peaks in a prime-time secondary bump ≈ 1.45× around hours 19–21 UTC, easing to 1.18× by hour 24. The table in `main.py` (`PARAM_RATIO`) is smoothed from the practice comparables' median ratios, so fallback and empirical modes share the same shape.
3. **Quantiles:** lognormal-style — `q_τ(h) = p1 · exp(log-ratio median + z_τ · σ_h)`, z = (−1.645, −0.674, 0, 0.674, 1.645). Spread `σ_h` widens with horizon: at least ≈ `0.15·√(h−1)` (the house baseline's walk), estimated from the comparables' inter-quantile range when available, clipped to [0.18, 1.25]. Non-median quantiles are also blended 50/50 with the (median-shifted) empirical quantiles for honest asymmetry. Output is enforced strictly non-decreasing (min log spacing 0.02), rounded to 2 decimals half-up like the starter, and clipped to [1, 1e9].

Deterministic (no randomness, no wall clock, no network), stdlib only, ~0.03 s per practice case locally — far under the 8 s / 1 CPU / 1024 MB limit.

## Run it

```sh
# Grader-style (practice kit cases):
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py   # writes out/<id>.out per <id>.in

# Local fallback modes:
echo '{"hour1": 2000000}' | python3 main.py        # simple JSON on stdin -> JSON on stdout
python3 main.py case.json                          # case file path in argv
```

Accepted fallback input shapes: a full `makefun/twitch-ccu-case/v1` case, `{"known": [{"hour": 1, "viewers": N}]}`, `{"hour1": N}` (also `hour_1` / `p1` / `first_hour`), a list of hourly rows/values, or a plain number. Missing input hours count as 0 viewers.

## Tests performed (2026-11-17)

- Grader-style run on all 4 practice kit cases (p00 Wolverine p1=23,357; p01 RDR2 p1=116,550; p02 God of War Ragnarök p1=269,517; p03 Spider-Man 2 p1=119,087): all outputs valid — exactly `{"hours": [...]}`, 23 rows hours 2–24, 5 quantiles each, strictly non-decreasing, all in (0, 1e9]. Total runtime 0.12 s for 4 cases.
- Synthetic stdin cases `hour1 = 2,000,000`, `500,000`, `0`, `{}` (missing), and the `known`-form of 2,000,000: all valid shape (0/missing use the documented fallback anchor).
- Determinism: two identical runs byte-identical (both stdin mode and practice cases).
- argv file-path mode verified.

## Files

- `main.py` — the entry (Python 3.13, stdlib only), at the folder root as the grader expects (`/bot/main.py` in the sandbox).
- `LICENSE` — MIT (an OSI-approved license is required at submission).
- `README.md` — this file.

## Submission (NOT done — requires the user)

Per the requirements API, submitting needs: a public GitHub repo owned by the user's GitHub account, created after 2026-11-17T06:00:00Z, with the make.fun README badge/branding, `repo_url` + full `commit_sha`, the AI-agent transcript bundle (kit CLI `node cli/makefun.mjs bundle`), and an entry signed within 10 minutes by a Solana wallet linked to the user's make.fun account (escrowed prize). None of that was done here: no sign-in, no account/wallet, no payment, no submission.

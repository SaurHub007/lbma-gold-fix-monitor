# LBMA Gold Fix vs Live XAU/USD

**Question:** The LBMA Gold Price is set twice a day (10:30 and 15:00 London). Does the live gold price treat that fix as a meaningful level — bouncing off it or drifting back to it — or is it just another number on the tape?

**Live dashboard:** https://saurhub007.github.io/lbma-gold-fix-monitor/

## How it works

A GitHub Actions job runs every 30 minutes during the gold trading week:

1. **Prices** — pulls 5-minute COMEX gold bars and appends them to `data/prices_5m.csv`, so the history keeps growing.
2. **Live spot** — snapshots XAU/USD spot and records the futures–spot gap (basis) to convert between the two.
3. **Fix levels** — the price at 10:30 (AM) and 15:00 (PM) London each day. Official LBMA prices added to `data/lbma_fixes.csv` replace the proxy for that day.
4. **Tests**, each against placebo levels (same method at 08:15 / 12:45 / 19:00 London, and at fix ±0.25–1.0%):
   - **Respect test** — when price comes back to the level, how often does it bounce vs break? A real level should bounce more often than the placebos (two-proportion z-test).
   - **Magnet test** — regress the next 2-hour move on distance from the level. A negative, significant beta means price gets pulled back toward the fix.
5. Writes `docs/data/summary.json`, which the GitHub Pages dashboard reads. The dashboard also streams the live spot price in the browser.

## Adding official LBMA prices

The official LBMA price data is licensed, so it's not scraped. To use the official numbers, add rows to `data/lbma_fixes.csv` (USD/oz):

```
date,am,pm
2026-11-06,<AM price>,<PM price>
```

Committing the file triggers a re-run.

## Stack
Python (pandas, numpy) · GitHub Actions · GitHub Pages · Chart.js

*Research project. Not investment advice.*

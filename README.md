# 2026 FIFA World Cup predictor

A small, self-contained model that predicts every 2026 FIFA World Cup match — win / draw / loss probabilities, the most likely top scorer per team, and a chart for each game. Re-runs pick up the latest results, retrain on the fresh data, refresh the upcoming predictions, and score the previous run's calls.

```
python predict_today.py                   # predict every still-upcoming fixture
python predict_today.py "Spain" "Brazil"  # predict a single match
python predict_today.py --refresh         # force a fresh pull of upstream data
```

## What it does

On every run, in order:

1. **Refreshes data.** Pulls the latest `results.csv` and `goalscorers.csv` from the [martj42/international_results](https://github.com/martj42/international_results) dataset. Cached locally and re-downloaded automatically when older than 24 hours (or immediately with `--refresh`).
2. **Scores the previous run.** Walks every `predictions/<run>/predictions.csv` from earlier sessions and, for any match that has now been played, fills in the actual score, marks the pick correct or not, and computes a per-match log-loss. Prints a scoreboard of newly-scored matches plus lifetime accuracy and log-loss across every prediction ever made.
3. **Trains the model** (XGBoost classifier) on all international matches since 2006 up to today.
4. **Predicts every still-upcoming fixture** in `data_cache/fixtures.csv`, skipping matches already played and knockout slots that are still placeholders (e.g. "Winner Group A").
5. **Predicts the top scorer for each team** in every match — one player per side.
6. **Writes** a per-match chart and a single `predictions.csv` summary under `predictions/<today>/`.

## How the match model works

For each match the model builds:

- **Elo ratings.** Recomputed from scratch over every international result since 2006. Each team starts at 1500 and trades points after every game, with bigger swings for blowouts and upsets.
- **Recent form.** Win rate and goal difference over each team's last 5 and 10 matches.
- **Rest days** since each team's last fixture.
- **Head-to-head.** How these two teams have done against each other historically (sample size, win rate, goal difference).
- **Context flags.** Neutral venue and tournament weight (a World Cup match counts for more than a friendly).

Those go into an **XGBoost** classifier that outputs win / draw / loss probabilities. To remove which-team-is-listed-as-home bias the model is evaluated both ways and the two predictions averaged. The output gets a tag — `LOCK` (≥60% on the pick), `LEAN` (≥45%), or `TOSS-UP`, with a `⚠️ UPSET PICK` flag when the model favors the team with the lower Elo.

On a held-out validation set (matches from 2023 onward) it currently lands around **60% accuracy** with a log-loss of **~0.86 vs. ~1.05** for a no-skill baseline.

## How the top-scorer pick works

For each team in a match:

1. Take every (non-own-goal) international goal scored for that national team in the last ~4 years.
2. Weight each goal by an exponential decay with a **540-day half-life** so recent goals dominate.
3. The player with the highest weighted goal sum is the pick. Their *share* of the team's recent weighted goals becomes the player share.
4. Combine with the team's recent goals-per-match to get an expected-goals estimate for that player in this match.
5. Convert to "scores at least one" via a Poisson assumption: `p = 1 - exp(-xG)`.

Results are printed and saved as `home_top_scorer`, `home_scorer_xg`, `home_scorer_prob` (and the same for `away_*`) in `predictions.csv`.

## How the scoring layer works

Each prediction CSV gets five extra columns the first time the match is played:

- `actual_home_score`, `actual_away_score`
- `actual_outcome` — `home` / `draw` / `away`
- `correct` — did the pick match the actual outcome
- `match_logloss` — `-log(p_actual)` for the model's probability on the realized outcome

The console prints both a "newly scored since last run" block and a lifetime total.

## Setup

```
pip install -r requirements.txt
python predict_today.py
```

Requirements: pandas, numpy, requests, xgboost, scikit-learn, matplotlib.

Team order in single-match mode doesn't matter, and common spellings work (typing `Iran` is fine even though the fixture file lists `IR Iran`). Run it with no arguments and a single team for an interactive prompt.

## CLI flags

| flag                  | what it does                                                          |
|-----------------------|------------------------------------------------------------------------|
| *(no args)*           | Predict every upcoming fixture; score previous predictions.            |
| `--all`, `-a`         | Same as no args; forces all-fixtures mode if a single team is given.   |
| `--refresh`           | Force-refresh `results.csv` and `goalscorers.csv` from upstream.       |
| `"Team A" "Team B"`   | Predict a single match between the two named teams.                    |

## Output

```
predictions/
└── 2026-06-24/
    ├── predictions.csv         # one row per match + actual scores once played
    ├── viz_Spain_vs_Brazil.png
    ├── viz_Norway_vs_France.png
    └── ...
```

## What it doesn't do (yet)

Honest limits matter more than the accuracy number:

- The team model rates *teams*, not the eleven players actually on the pitch — no injuries, suspensions, lineups, manager or tactics.
- No expected goals (xG) data — the stat that actually moves modern soccer models.
- No "this team only needs a draw to advance" tournament context.
- The top-scorer pick uses historical international scorers in the last 4 years; it can't tell that a player has retired, dropped out of the squad, or is suspended. Club goals don't count. A team's recent goals-per-match is averaged across opposition strength, so the absolute probabilities are approximate — treat them as a relative ranking.
- Models under-call draws, because draws don't have a clean statistical fingerprint.
- Knockout fixtures only get predicted once the bracket placeholders in `fixtures.csv` are replaced with actual team names.

Judge the model over a slate of games with log-loss, not on any single result.

## Data

- Historical international results and goalscorers: [martj42/international_results](https://github.com/martj42/international_results) (refreshed each run from `results.csv` and `goalscorers.csv`).
- Fixtures: official 2026 World Cup schedule, in `data_cache/fixtures.csv`.

## Credits

Thanks to Mariana Antaya for the base of this project — the team-match prediction skeleton (Elo + form + H2H + XGBoost, the per-match chart style, and most of the original `predict_today.py`) comes from her [world_cup_predictions](https://github.com/mar-antaya/world_cup_predictions) repo. This fork adds:

- Auto-refresh of the upstream results data.
- An all-fixtures mode that predicts every upcoming match per run.
- A self-scoring layer that grades previous predictions against the latest results.
- A goalscorer layer that picks one likely top scorer per team using `goalscorers.csv`.

## License

MIT — same as the upstream repo.

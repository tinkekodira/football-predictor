"""Who is missing, using only what was knowable before kick-off.

The goals model estimates a team's strength from its recent results. It has no
idea whether the striker who scored most of those goals is injured today, and
that is a real gap: the handoff calls it a structural disadvantage no
hyperparameter fixes.

**The whole difficulty of this module is that the obvious version cheats.**
Understat records who actually played. Confirmed line-ups are published about
an hour before kick-off - after the opening price this project bets into, and
at or after the closing line it measures against. A model told who played would
be reading the future, and because availability genuinely matters it would not
look like a bug. It would look like a large, stable, thoroughly convincing
edge, which is the worst thing a bug can look like.

So nothing here reads the match it describes. Every feature for a match on date
D is computed from that team's matches strictly before D, and
`test_availability.py` asserts it by rewriting the match being described and
checking the numbers do not move.

**What the features mean.**

`missing_share` is the share of a team's recent on-pitch minutes belonging to
players who did not appear in its most recent match. It deliberately measures
*newly* missing players rather than everyone unavailable. A player injured for
three months has already been absent from the results the model learned its
strength from, so that absence is priced in; counting it again would be double
counting. Someone who played every week until last Saturday is the one the
strength estimate is still wrong about.

That is also why importance and absence are read from *different* windows.
Importance comes from the ten matches before last; absence from the most recent
match alone. Overlapping them would let a single missed match quietly reduce
the importance of the very player whose absence is being measured.

`missing_starter_share` restricts that to players who usually started, which is
a cleaner signal at the cost of a smaller one: a fringe substitute dropping out
says little about how strong the team is today.

`missing_xgchain_share` weights players by their share of the team's recent
xGChain instead of its minutes. Minutes are the wrong currency for a question
about scoring: a full-back and a centre-forward play the same ninety, and
losing them does not cost the same number of goals. xGChain credits every
player involved in a move that ended in a shot, so it is a reasonable measure
of how much of a team's attacking output ran through someone. The trade is that
it says almost nothing about defenders, so it is the right weight for the
attacking side of a fixture and the wrong one for the defending side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# How many matches back to measure a player's importance to the team.
# Ten is roughly a third of a season: long enough that one rested week does not
# swing it, short enough to follow a change in who actually plays.
DEFAULT_IMPORTANCE_WINDOW = 10

# How many recent matches an absence is read from. One is the sharpest signal
# for "something has changed"; two is steadier but blurs the line with rotation.
DEFAULT_ABSENCE_WINDOW = 1

# Below this many prior matches the features are left missing rather than
# guessed. Early in a team's first season there is no baseline to be missing
# from, and a fabricated zero would read as "full strength available".
MIN_PRIOR_MATCHES = 6

# A team fields eleven players for ninety minutes.
TEAM_MINUTES_PER_MATCH = 11 * 90


def team_match_view(lineups: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per appearance, labelled with the team and the date.

    Splits each match's line-up rows onto the team that owns them, so the rest
    of this module can work team by team without carrying home and away around.
    """
    fixtures = matches[["match_id", "date", "home_team", "away_team"]].copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    merged = lineups.merge(fixtures, on="match_id", how="inner")
    merged["team"] = np.where(
        merged["is_home"].astype(bool), merged["home_team"], merged["away_team"]
    )
    return merged


def availability_features(
    lineups: pd.DataFrame,
    matches: pd.DataFrame,
    importance_window: int = DEFAULT_IMPORTANCE_WINDOW,
    absence_window: int = DEFAULT_ABSENCE_WINDOW,
    min_prior_matches: int = MIN_PRIOR_MATCHES,
) -> pd.DataFrame:
    """Per-team availability for every match, from strictly earlier matches.

    One row per (match_id, is_home), with NaN wherever there is too little
    history to say anything honest rather than a zero that would read as a
    fully fit squad.
    """
    columns = [
        "match_id", "is_home", "team", "missing_share",
        "missing_starter_share", "missing_xgchain_share", "n_missing", "n_prior",
    ]
    view = team_match_view(lineups, matches)
    if view.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for team, block in view.groupby("team", sort=False):
        ordered = _squads_in_order(block)
        for position in range(len(ordered)):
            rows.append(
                _features_at(
                    ordered, position, team,
                    importance_window, absence_window, min_prior_matches,
                )
            )
    return pd.DataFrame(rows, columns=columns)


def _squads_in_order(block: pd.DataFrame) -> list[dict]:
    """One team's matches oldest first, each with who played and who started."""
    ordered = []
    grouped = block.groupby(["date", "match_id", "is_home"], sort=True)
    for (_date, match_id, is_home), group in grouped:
        minutes = dict(
            zip(group["player_id"].astype(str), group["minutes"].astype(float))
        )
        chain = (
            dict(zip(group["player_id"].astype(str), group["xgchain"].astype(float)))
            if "xgchain" in group.columns else {}
        )
        starters = set(
            group.loc[group["started"].astype(bool), "player_id"].astype(str)
        )
        ordered.append(
            {
                "match_id": match_id,
                "is_home": bool(is_home),
                "minutes": minutes,
                "chain": chain,
                "starters": starters,
            }
        )
    return ordered


def _features_at(
    ordered: list[dict],
    position: int,
    team: str,
    importance_window: int,
    absence_window: int,
    min_prior_matches: int,
) -> dict:
    """Availability for one team-match, reading only entries before `position`."""
    blank = {
        "match_id": ordered[position]["match_id"],
        "is_home": ordered[position]["is_home"],
        "team": team,
        "missing_share": float("nan"),
        "missing_starter_share": float("nan"),
        "missing_xgchain_share": float("nan"),
        "n_missing": float("nan"),
        "n_prior": position,
    }
    if position < min_prior_matches:
        return blank

    # These two slices are the point-in-time rule made concrete. Neither ever
    # includes `position`, so the match being described cannot contribute to
    # its own features, and they do not overlap each other, so a missed match
    # cannot deflate the importance of the player who missed it.
    absence_start = max(0, position - absence_window)
    absence_rows = ordered[absence_start:position]
    importance_start = max(0, absence_start - importance_window)
    importance_rows = ordered[importance_start:absence_start]
    if not absence_rows or not importance_rows:
        return blank

    minutes: dict[str, float] = {}
    chain: dict[str, float] = {}
    starts: dict[str, int] = {}
    for entry in importance_rows:
        for player, played in entry["minutes"].items():
            minutes[player] = minutes.get(player, 0.0) + played
        for player, value in entry.get("chain", {}).items():
            if value == value:  # skip NaN without importing numpy here
                chain[player] = chain.get(player, 0.0) + value
        for player in entry["starters"]:
            starts[player] = starts.get(player, 0) + 1

    total = TEAM_MINUTES_PER_MATCH * len(importance_rows)
    if total <= 0:
        return blank

    appeared: set[str] = set()
    for entry in absence_rows:
        appeared |= set(entry["minutes"])

    missing = [player for player in minutes if player not in appeared]
    regular = [p for p in missing if starts.get(p, 0) * 2 > len(importance_rows)]

    chain_total = sum(chain.values())
    chain_share = (
        sum(chain.get(p, 0.0) for p in missing) / chain_total
        if chain_total > 0 else float("nan")
    )

    return blank | {
        "missing_share": sum(minutes[p] for p in missing) / total,
        "missing_starter_share": sum(minutes[p] for p in regular) / total,
        "missing_xgchain_share": chain_share,
        "n_missing": float(len(missing)),
    }


def attach_to_fixtures(features: pd.DataFrame) -> pd.DataFrame:
    """Fold the per-team rows into one row per match, home and away together."""
    if features.empty:
        return pd.DataFrame()
    columns = [
        "missing_share", "missing_starter_share", "missing_xgchain_share",
        "n_missing", "n_prior",
    ]
    home = features[features["is_home"].astype(bool)].set_index("match_id")[columns]
    away = features[~features["is_home"].astype(bool)].set_index("match_id")[columns]
    joined = home.join(away, lsuffix="_home", rsuffix="_away", how="inner")
    return joined.reset_index()

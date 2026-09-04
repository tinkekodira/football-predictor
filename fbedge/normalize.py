"""Turn raw football-data.co.uk CSVs into a canonical schema.

Three jobs live here:

1. **Column mapping.** The raw files use cryptic short names (FTHG, HST, HC)
   and, importantly, *not every file has every column*. Referee, shots and
   corners are missing from some leagues and from most pre-2005 seasons. So
   every column is treated as optional and missing ones become nulls rather
   than crashes.

2. **Team names.** The source is mostly self-consistent, but spellings drift
   across seasons and will certainly not match Understat or FBref when those
   are added in Phase 5. Everything is resolved through one alias table.

3. **Derived metrics.** The market-facing quantities (BTTS, totals, cards)
   are computed once, here, so that the model layer and the descriptive layer
   can never disagree about what "total cards" means.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

from . import config

# --------------------------------------------------------------------------
# Column mapping
# --------------------------------------------------------------------------

# raw column -> canonical column. Anything not listed is dropped from the
# matches table (odds columns are handled separately, below).
CORE_COLUMNS: dict[str, str] = {
    "Date": "date_raw",
    "Time": "kickoff_time",
    "HomeTeam": "home_team_raw",
    "AwayTeam": "away_team_raw",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "HTR": "result_ht",
    "Referee": "referee_raw",
    "Attendance": "attendance",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HO": "home_offsides",
    "AO": "away_offsides",
    "HHW": "home_woodwork",
    "AHW": "away_woodwork",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

# A few files use "HG"/"AG"/"Res" instead of "FTHG"/"FTAG"/"FTR".
#
# HFKC/AFKC are free kicks conceded, which the source substitutes for fouls in
# the handful of competitions where fouls are not reported separately. They
# count offsides and other offences as well, so they run slightly higher than
# fouls; using them is better than having no discipline signal at all, and the
# card model absorbs the difference into the league intercept.
COLUMN_FALLBACKS: dict[str, str] = {
    "HG": "home_goals",
    "AG": "away_goals",
    "Res": "result",
    "HFKC": "home_fouls",
    "AFKC": "away_fouls",
}

INT_COLUMNS = [
    "home_goals", "away_goals", "home_goals_ht", "away_goals_ht",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_offsides", "away_offsides",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
    "home_woodwork", "away_woodwork", "attendance",
]


# --------------------------------------------------------------------------
# Odds specifications
# --------------------------------------------------------------------------
# "open" is the price when the market was first published, "close" is the last
# price before kick-off. Closing prices are the ones that matter: the closing
# line is the most accurate public probability estimate that exists, so it is
# both the benchmark the models must beat and the basis for measuring closing
# line value in Phase 3.
#
# Column names drifted over the years (Pinnacle was "PH/PD/PA" before it
# became "PSH/PSD/PSA"; the market max/average used to be Betbrain's "BbMx"/
# "BbAv"). Specs whose columns are absent from a given file are skipped.

# Bookmakers, keyed by the column prefix the source uses. Verified against
# football-data.co.uk/notes.txt. Closing prices are the same prefix with a "C"
# inserted before the outcome letter, so B365H becomes B365CH.
#
# Betfair Exchange earns its place at the top: an exchange charges commission
# rather than building a margin into the price, so its overround is a fraction
# of a bookmaker's. That makes it the best closing-line benchmark in this file
# for anyone who cannot get on at Pinnacle.
BOOKMAKERS: dict[str, str] = {
    "BFE": "betfair_exchange",
    "PS": "pinnacle",
    "P": "pinnacle",                 # older files used the shorter prefix
    "Max": "market_max",             # best price available anywhere
    "Avg": "market_avg",             # market consensus
    "B365": "bet365",
    "1XB": "1xbet",
    "BMGM": "betmgm",
    "BV": "betvictor",
    "WH": "william_hill",
    "LB": "ladbrokes",
    "PP": "paddy_power",
    "CL": "coral",
    "SKB": "skybet",
    "BF": "betfair_sportsbook",
    "BFD": "betfred",
    "IW": "interwetten",
    "BW": "bet_and_win",
    "VC": "vc_bet",
    "SB": "sportingbet",
    "SJ": "stan_james",
    "SY": "stanleybet",
    "SO": "sporting_odds",
    "BS": "blue_square",
    "GB": "gamebookers",
}

# Which prefixes ever carried totals and handicap prices. The rest are 1X2
# only, and generating columns for them would just be dead weight.
#
# **BFE was missing from both lists until 2026-09-04, and it mattered.** The
# source has carried `BFE>2.5`/`BFE<2.5` and `BFEAHH`/`BFEAHA` since 2024/25,
# and Betfair Exchange heads `backtest.FAIR_LINE_PREFERENCE` precisely because
# an exchange's overround is a fraction of a bookmaker's. Leaving it out of
# these two tuples meant the sharpest available benchmark was silently dropped
# for totals and handicaps while being used for 1X2, so a single backtest
# measured two markets against two different instruments. See BACKLOG B10.
TOTALS_PREFIXES = ("P", "B365", "Max", "Avg", "GB", "BFE")
HANDICAP_PREFIXES = ("P", "B365", "Max", "Avg", "GB", "LB", "BFE")

# Bookmaker-specific handicap line columns, preferred over the market-wide
# AHh when present, because a book's own line is what its prices refer to.
HANDICAP_LINE_COLUMNS = {"B365": "B365AH", "GB": "GBAH", "LB": "LBAH"}


def _build_1x2_specs() -> list[tuple[str, str, str, str, str]]:
    """(bookmaker, phase, home_col, draw_col, away_col) for every book."""
    specs = []
    for prefix, name in BOOKMAKERS.items():
        specs.append((name, "open", f"{prefix}H", f"{prefix}D", f"{prefix}A"))
        specs.append((name, "close", f"{prefix}CH", f"{prefix}CD", f"{prefix}CA"))
    # Betbrain's market aggregates, retired but present in older files.
    specs.append(("market_max", "open", "BbMxH", "BbMxD", "BbMxA"))
    specs.append(("market_avg", "open", "BbAvH", "BbAvD", "BbAvA"))
    return specs


def _build_totals_specs() -> list[tuple[str, str, float, str, str]]:
    """(bookmaker, phase, line, over_col, under_col). Only 2.5 is published."""
    specs = []
    for prefix in TOTALS_PREFIXES:
        name = BOOKMAKERS[prefix]
        specs.append((name, "open", 2.5, f"{prefix}>2.5", f"{prefix}<2.5"))
        specs.append((name, "close", 2.5, f"{prefix}C>2.5", f"{prefix}C<2.5"))
    specs.append(("market_max", "open", 2.5, "BbMx>2.5", "BbMx<2.5"))
    specs.append(("market_avg", "open", 2.5, "BbAv>2.5", "BbAv<2.5"))
    return specs


def _build_handicap_specs() -> list[tuple[str, str, str, str, str]]:
    """(bookmaker, phase, line_col, home_col, away_col).

    The market-wide handicap column is AHh from 2019/20 onwards and BbAHh
    before that; a few books also publish their own line, which takes
    precedence because their prices refer to it.
    """
    specs = []
    for prefix in HANDICAP_PREFIXES:
        name = BOOKMAKERS[prefix]
        own_line = HANDICAP_LINE_COLUMNS.get(prefix)
        for line_col in filter(None, (own_line, "AHh", "BbAHh")):
            specs.append((name, "open", line_col, f"{prefix}AHH", f"{prefix}AHA"))
        specs.append((name, "close", "AHCh", f"{prefix}CAHH", f"{prefix}CAHA"))
    specs.append(("market_max", "open", "BbAHh", "BbMxAHH", "BbMxAHA"))
    specs.append(("market_avg", "open", "BbAHh", "BbAvAHH", "BbAvAHA"))
    return specs


ODDS_1X2 = _build_1x2_specs()
ODDS_TOTALS = _build_totals_specs()
ODDS_ASIAN_HANDICAP = _build_handicap_specs()


# --------------------------------------------------------------------------
# Team names
# --------------------------------------------------------------------------

# Alternative spellings -> the spelling used as canonical. The canonical form
# is football-data.co.uk's own, because that is the primary source; the aliases
# exist so that a user typing "Manchester United" in the app finds the team,
# and so that Phase 5 joins against other providers have somewhere to live.
TEAM_ALIASES: dict[str, str] = {
    # England
    "manchester united": "Man United",
    "man utd": "Man United",
    "manchester city": "Man City",
    "nottingham forest": "Nott'm Forest",
    "notts forest": "Nott'm Forest",
    "sheffield united": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "sheffield wednesday": "Sheffield Weds",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "newcastle united": "Newcastle",
    "west ham united": "West Ham",
    "leeds united": "Leeds",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "afc bournemouth": "Bournemouth",
    "leicester city": "Leicester",
    "norwich city": "Norwich",
    "cardiff city": "Cardiff",
    "swansea city": "Swansea",
    "stoke city": "Stoke",
    "hull city": "Hull",
    "ipswich town": "Ipswich",
    "coventry city": "Coventry",
    "luton town": "Luton",
    # Spain
    "atletico madrid": "Ath Madrid",
    "atlético madrid": "Ath Madrid",
    "atl madrid": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao",
    "athletic club": "Ath Bilbao",
    "real betis": "Betis",
    "espanyol": "Espanol",
    "rcd espanyol": "Espanol",
    "sporting gijon": "Sp Gijon",
    "rayo vallecano": "Vallecano",
    "deportivo la coruna": "La Coruna",
    "celta vigo": "Celta",
    "real sociedad": "Sociedad",
    "real valladolid": "Valladolid",
    "villarreal cf": "Villarreal",
    "sevilla fc": "Sevilla",
    "valencia cf": "Valencia",
    "girona fc": "Girona",
    # Italy
    "internazionale": "Inter",
    "inter milan": "Inter",
    "ac milan": "Milan",
    "as roma": "Roma",
    "ss lazio": "Lazio",
    "hellas verona": "Verona",
    "ssc napoli": "Napoli",
    "atalanta bc": "Atalanta",
    "us salernitana": "Salernitana",
    "ac monza": "Monza",
    # Germany
    "bayern munchen": "Bayern Munich",
    "bayern münchen": "Bayern Munich",
    "fc bayern munich": "Bayern Munich",
    "borussia dortmund": "Dortmund",
    "bvb": "Dortmund",
    "borussia monchengladbach": "M'gladbach",
    "borussia mönchengladbach": "M'gladbach",
    "monchengladbach": "M'gladbach",
    "gladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt",
    "bayer leverkusen": "Leverkusen",
    "bayer 04 leverkusen": "Leverkusen",
    "hertha berlin": "Hertha",
    "hertha bsc": "Hertha",
    "fc koln": "FC Koln",
    "koln": "FC Koln",
    "köln": "FC Koln",
    "cologne": "FC Koln",
    "rb leipzig": "RB Leipzig",
    "vfb stuttgart": "Stuttgart",
    "werder bremen": "Werder Bremen",
    "vfl wolfsburg": "Wolfsburg",
    "tsg hoffenheim": "Hoffenheim",
    "sc freiburg": "Freiburg",
    "fsv mainz": "Mainz",
    "mainz 05": "Mainz",
    "union berlin": "Union Berlin",
    # France
    "paris saint-germain": "Paris SG",
    "paris saint germain": "Paris SG",
    "psg": "Paris SG",
    "olympique marseille": "Marseille",
    "olympique de marseille": "Marseille",
    "olympique lyonnais": "Lyon",
    "as monaco": "Monaco",
    "lille osc": "Lille",
    "stade rennais": "Rennes",
    "ogc nice": "Nice",
    "rc lens": "Lens",
    "fc nantes": "Nantes",
    "stade brestois": "Brest",
    "montpellier hsc": "Montpellier",
    "rc strasbourg": "Strasbourg",
    "toulouse fc": "Toulouse",
}


def _is_missing(value) -> bool:
    """True for None, NaN, pd.NA and blank strings.

    A plain `isinstance(value, float) and isnan(value)` check is not enough:
    when a column is absent from a source file it is filled with `pd.NA`,
    which is neither None nor a float, and `str(pd.NA)` cheerfully produces
    the string "<NA>". That silently populated the referee column with a
    sentinel that looked like real data to every downstream query.
    """
    if value is None or value is pd.NA:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):  # arrays and unhashable objects
        return False
    return isinstance(value, str) and not value.strip()


def _strip_accents(text: str) -> str:
    """Remove diacritics so that 'Köln' and 'Koln' compare equal."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _alias_key(name: str) -> str:
    """Lookup key for the alias table: lowercase, unaccented, punctuation-free."""
    cleaned = _strip_accents(str(name)).lower()
    cleaned = cleaned.replace("'", "").replace("'", "").replace(".", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def canonical_team(name: str) -> str:
    """Resolve a team name to its canonical spelling.

    Unknown names pass through with whitespace tidied rather than being
    rejected, so a newly promoted club never breaks ingestion. Add it to
    TEAM_ALIASES only if a second spelling for it shows up.
    """
    if _is_missing(name):
        return ""
    tidy = re.sub(r"\s+", " ", str(name).replace("\u2019", "'")).strip()
    return TEAM_ALIASES.get(_alias_key(tidy), tidy)


def team_slug(name: str) -> str:
    """URL/ID-safe form of a team name: 'Nott'm Forest' -> 'nottm_forest'."""
    key = _strip_accents(canonical_team(name)).lower().replace("'", "")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", key)).strip("_")


def resolve_team(query: str, known_teams: list[str]) -> str | None:
    """Best-effort lookup of a user-typed name against teams in the database.

    Tried in order: canonical match, case-insensitive match, prefix match,
    substring match. Returns None when nothing matches or a prefix/substring
    search is ambiguous, so the caller can ask the user rather than guess.
    """
    if not query or not query.strip():
        return None

    canon = canonical_team(query)
    if canon in known_teams:
        return canon

    lowered = {t.lower(): t for t in known_teams}
    if canon.lower() in lowered:
        return lowered[canon.lower()]

    needle = _alias_key(canon)
    for matcher in (
        lambda t: _alias_key(t).startswith(needle),
        lambda t: needle in _alias_key(t),
    ):
        hits = [t for t in known_teams if matcher(t)]
        if len(hits) == 1:
            return hits[0]

    # Last resort: match the query against the alias table itself, so that a
    # partial common name ("atletico") reaches a canonical form the source
    # spells quite differently ("Ath Madrid").
    via_alias = {
        target
        for alias, target in TEAM_ALIASES.items()
        if (alias.startswith(needle) or needle in alias) and target in known_teams
    }
    return via_alias.pop() if len(via_alias) == 1 else None


# Corporate and legal noise that carries no information about which club is
# meant. Stripped from the *ends* of a name before comparing, so "FC Augsburg"
# and "Augsburg" are one club while a club actually called "Milan" is not eaten
# by the "AC" rule applied in the middle. Ordered longest-first so "1. FC" is
# tried before "FC".
#
# Lifted out of `injuries.py`, where it was written for API-Football and then
# needed verbatim for the openfootball calendar: 72 of 96 club names in that
# feed differ from this project's only by a suffix like "FC" or "AFC".
EXTERNAL_NAME_NOISE = (
    "football club", "fussball club", "association", "calcio", "balompie",
    "1. fc", "1.fc", "1 fc", "afc", "fsv", "tsg", "vfl", "vfb", "sv", "sc",
    "fc", "cf", "ac", "as", "ss", "us", "rc", "sd", "ud", "cp", "cd", "aj",
    "sco", "cfc", "rcd", "ca", "de", "1907", "1909", "1913", "1901", "07",
    "05", "04", "29", "1899",
)


def external_name_key(name: str) -> str:
    """A comparable form of a club name from somebody else's database.

    Accents removed, punctuation dropped, corporate prefixes and suffixes
    stripped, spacing collapsed. The point is only to make two spellings of one
    club compare equal. **It is never used to decide that two different strings
    are the same club**, which is what a similarity score would do - and which
    the Understat integration established would silently merge Milan with
    Inter.
    """
    text = _strip_accents(str(name)).lower()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    words = text.split()
    changed = True
    while changed and words:
        changed = False
        for noise in EXTERNAL_NAME_NOISE:
            parts = noise.split()
            if len(words) > len(parts) and words[: len(parts)] == parts:
                words, changed = words[len(parts):], True
                break
            if len(words) > len(parts) and words[-len(parts):] == parts:
                words, changed = words[: -len(parts)], True
                break
    return " ".join(words)


def resolve_external_team(
    name: str, known: set[str] | list[str], aliases: dict[str, str] | None = None
) -> str | None:
    """Map another provider's club name onto this project's, or return None.

    Four attempts, in decreasing confidence: the name as given; this module's
    own alias table; a caller-supplied alias table for that provider's known
    quirks; and a comparison after stripping corporate noise from both sides.

    **No fuzzy fallback, and None is a real answer.** The caller reports it and
    drops the row. A calendar with a silently unmapped team is a fixture that
    never joins to anything, which looks exactly like a fixture that was never
    scheduled - and this project has paid for that shape of bug before.
    """
    known = set(known)
    if name in known:
        return name

    direct = canonical_team(name)
    if direct in known:
        return direct

    for table in (aliases or {}, {}):
        mapped = table.get(external_name_key(name))
        if mapped and mapped in known:
            return mapped

    key = external_name_key(name)
    if not key:
        return None
    for candidate in known:
        if external_name_key(candidate) == key:
            return candidate
    # This module's own alias table, tried both as written and with the
    # corporate noise stripped. "Manchester City FC" is not a key in it;
    # "manchester city" is, and dropping the suffix is what finds it.
    for lookup in (_alias_key(name), key):
        via_alias = TEAM_ALIASES.get(lookup)
        if via_alias in known:
            return via_alias
    return None


def canonical_referee(name: str) -> str | None:
    """Tidy a referee name. Returns None for blanks so SQL sees NULL.

    The source writes referees as 'M Oliver' / 'M. Oliver' / 'Oliver M' with
    inconsistent spacing. Full disambiguation needs a proper referee table,
    which is Phase 5 work; for now we only normalise punctuation and spacing.
    """
    if _is_missing(name):
        return None
    tidy = re.sub(r"\s+", " ", str(name).replace(".", " ")).strip()
    return tidy or None


# --------------------------------------------------------------------------
# Dates and identifiers
# --------------------------------------------------------------------------

def parse_dates(raw: pd.Series) -> pd.Series:
    """Parse football-data dates, which are dd/mm/yy in older files and
    dd/mm/yyyy in newer ones - sometimes both within a single league's run.
    """
    text = raw.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna() & text.notna()
    if missing.any():
        parsed = parsed.fillna(
            pd.to_datetime(text.where(missing), format="%d/%m/%y", errors="coerce")
        )
    missing = parsed.isna() & text.notna()
    if missing.any():
        parsed = parsed.fillna(
            pd.to_datetime(text.where(missing), dayfirst=True, errors="coerce")
        )
    return parsed


def make_match_id(league: str, date: pd.Timestamp, home: str, away: str) -> str:
    """Deterministic, readable match identifier.

    Readable rather than a bare hash so that debugging a bad row does not
    require a database lookup. The short hash suffix guarantees uniqueness in
    the pathological case of a fixture being played twice on one day.
    """
    date_str = "unknown" if pd.isna(date) else pd.Timestamp(date).strftime("%Y%m%d")
    stem = f"{league}_{date_str}_{team_slug(home)}_{team_slug(away)}"
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()[:6]
    return f"{stem}_{digest}"


# --------------------------------------------------------------------------
# Main entry points
# --------------------------------------------------------------------------

def normalize_matches(
    raw: pd.DataFrame, league: str, season_start_year: int
) -> pd.DataFrame:
    """Map one raw league-season CSV onto the canonical `matches` schema."""
    df = raw.copy().reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    mapping = {src: dst for src, dst in CORE_COLUMNS.items() if src in df.columns}
    for src, dst in COLUMN_FALLBACKS.items():
        if src in df.columns and dst not in mapping.values():
            mapping[src] = dst

    out = df[list(mapping)].rename(columns=mapping)
    # Remember which raw row each canonical row came from: extract_odds needs
    # this to line odds up with the matches that survived filtering.
    out["source_row"] = out.index

    # Guarantee every canonical column exists, so downstream SQL never has to
    # care which league or season a row came from.
    for column in set(CORE_COLUMNS.values()) | set(COLUMN_FALLBACKS.values()):
        if column not in out.columns:
            out[column] = pd.NA

    out["date"] = parse_dates(out["date_raw"])
    out["home_team"] = out["home_team_raw"].map(canonical_team)
    out["away_team"] = out["away_team_raw"].map(canonical_team)
    out["referee"] = out["referee_raw"].map(canonical_referee)

    # Drop rows the source leaves behind as padding: no date, no teams, or an
    # unplayed fixture sitting in the current season's file.
    out = out[
        out["date"].notna()
        & (out["home_team"] != "")
        & (out["away_team"] != "")
        & out["home_goals"].notna()
        & out["away_goals"].notna()
    ].copy()

    for column in INT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    out["league"] = league
    out["league_name"] = config.LEAGUES.get(league, league)
    out["country"] = config.LEAGUE_COUNTRY.get(league)
    out["season_start_year"] = season_start_year
    out["season"] = config.season_label(season_start_year)
    out["match_id"] = [
        make_match_id(league, d, h, a)
        for d, h, a in zip(out["date"], out["home_team"], out["away_team"])
    ]

    out = _add_derived_metrics(out)

    keep = [
        "source_row",
        "match_id", "league", "league_name", "country",
        "season_start_year", "season", "date", "kickoff_time",
        "home_team", "away_team", "referee", "attendance",
        "home_goals", "away_goals", "result",
        "home_goals_ht", "away_goals_ht", "result_ht",
        "home_shots", "away_shots", "home_sot", "away_sot",
        "home_corners", "away_corners", "home_fouls", "away_fouls",
        "home_offsides", "away_offsides", "home_woodwork", "away_woodwork",
        "home_yellows", "away_yellows", "home_reds", "away_reds",
        "total_goals", "goal_difference", "btts",
        "over_1_5", "over_2_5", "over_3_5",
        "goals_ht", "goals_2h",
        "total_corners", "total_cards", "total_yellows", "total_reds",
        "booking_points", "total_shots", "total_sot", "total_fouls",
    ]
    return out[keep].sort_values(["date", "league", "home_team"]).reset_index(drop=True)


def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the market-facing quantities once, in one place."""
    home_goals = df["home_goals"].astype("Int64")
    away_goals = df["away_goals"].astype("Int64")

    df["total_goals"] = home_goals + away_goals
    df["goal_difference"] = home_goals - away_goals
    df["btts"] = (home_goals > 0) & (away_goals > 0)

    for line in (1.5, 2.5, 3.5):
        df[f"over_{str(line).replace('.', '_')}"] = df["total_goals"] > line

    df["goals_ht"] = df["home_goals_ht"] + df["away_goals_ht"]
    df["goals_2h"] = df["total_goals"] - df["goals_ht"]

    df["total_corners"] = df["home_corners"] + df["away_corners"]
    df["total_yellows"] = df["home_yellows"] + df["away_yellows"]
    df["total_reds"] = df["home_reds"] + df["away_reds"]
    df["total_cards"] = df["total_yellows"] + df["total_reds"]

    # The source's own bookings-points convention: 10 per yellow, 25 per red.
    # Most "total cards" betting markets settle on a similar weighting, so it
    # is worth carrying alongside the raw count.
    #
    # A counting inconsistency to be aware of, straight from the source notes:
    # in England and Scotland a first yellow is *not* recorded when a second
    # turns it into a red, whereas European competitions record both. English
    # card totals therefore run slightly lower than continental ones for the
    # same on-pitch events. Models are fitted per league, so the difference is
    # absorbed by the league intercept, but the raw numbers are not directly
    # comparable across countries.
    df["booking_points"] = df["total_yellows"] * 10 + df["total_reds"] * 25

    df["total_shots"] = df["home_shots"] + df["away_shots"]
    df["total_sot"] = df["home_sot"] + df["away_sot"]
    df["total_fouls"] = df["home_fouls"] + df["away_fouls"]
    return df


def extract_odds(raw: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide odds columns into a long, extensible table.

    Long format costs a join but means adding a bookmaker or a market in a
    later phase is a data change, not a schema migration.
    """
    if "source_row" not in matches.columns:
        raise ValueError(
            "extract_odds needs the 'source_row' column produced by "
            "normalize_matches; pass the frame through unmodified."
        )

    df = raw.copy().reset_index(drop=True)

    # Select exactly the raw rows that survived filtering, in the same order
    # as the canonical matches frame.
    df = df.loc[matches["source_row"].to_numpy()].reset_index(drop=True)
    return odds_long(df, matches["match_id"].reset_index(drop=True))


def odds_long(
    raw: pd.DataFrame, keys: pd.Series, key_name: str = "match_id"
) -> pd.DataFrame:
    """Reshape wide odds columns into long rows, keyed by whatever you supply.

    Split out of `extract_odds` so the Phase 4 fixture snapshots can reuse it
    verbatim. `fixtures.csv` uses exactly the same column conventions as a
    season file but has no results and therefore no `match_id`, so its rows are
    keyed by a content hash instead. Two readers of the same wide format would
    drift apart - the away-handicap sign alone is a bug this project has
    already paid for once - so there is only one.

    `raw` must already be row-aligned with `keys`.
    """
    df = raw.copy().reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    keys = pd.Series(keys).reset_index(drop=True)

    records: list[pd.DataFrame] = []

    def add(bookmaker, phase, market, selection, price_col, line=None, line_col=None,
            negate_line=False):
        if price_col not in df.columns:
            return
        # A handicap price whose line column is absent is not a price: nothing
        # can settle "home at 1.95" without knowing the start. Storing it with
        # a NULL line put ~98,500 unsettleable rows into the odds table, about
        # a third of it - `price_selection` returns None for them, so they were
        # dead weight rather than a wrong number, but a spec asking for a line
        # column the file does not have should produce nothing. See BACKLOG B11.
        if line_col is not None and line_col not in df.columns:
            return
        price = pd.to_numeric(df[price_col], errors="coerce")
        frame = pd.DataFrame(
            {
                key_name: keys,
                "bookmaker": bookmaker,
                "phase": phase,
                "market": market,
                "selection": selection,
                "price": price,
            }
        )
        if line_col is not None and line_col in df.columns:
            values = pd.to_numeric(df[line_col], errors="coerce")
            frame["line"] = -values if negate_line else values
        else:
            frame["line"] = line
        records.append(frame[price.notna()])

    for bookmaker, phase, home_col, draw_col, away_col in ODDS_1X2:
        add(bookmaker, phase, "1x2", "home", home_col)
        add(bookmaker, phase, "1x2", "draw", draw_col)
        add(bookmaker, phase, "1x2", "away", away_col)

    for bookmaker, phase, line, over_col, under_col in ODDS_TOTALS:
        add(bookmaker, phase, "total_goals", "over", over_col, line=line)
        add(bookmaker, phase, "total_goals", "under", under_col, line=line)

    for bookmaker, phase, line_col, home_col, away_col in ODDS_ASIAN_HANDICAP:
        add(bookmaker, phase, "asian_handicap", "home", home_col, line_col=line_col)
        # The source records one handicap column, applied to the home team.
        # The away price is for the *opposite* handicap: if the line is -0.75
        # on the home side it is +0.75 on the away side. Storing the raw column
        # against both selections would mean settling away bets on the wrong
        # line, so it is negated here.
        add(
            bookmaker, phase, "asian_handicap", "away", away_col,
            line_col=line_col, negate_line=True,
        )

    columns = [key_name, "bookmaker", "phase", "market", "selection", "line", "price"]
    if not records:
        return pd.DataFrame(columns=columns)

    odds = pd.concat(records, ignore_index=True)
    odds = odds[odds["price"] > 1.0]

    # A file may carry both the old and new Pinnacle column names; keep one row
    # per (key, bookmaker, phase, market, selection, line).
    odds = odds.drop_duplicates(
        subset=[key_name, "bookmaker", "phase", "market", "selection", "line"],
        keep="first",
    )
    return odds[columns].reset_index(drop=True)


def normalize_league_season(
    raw: pd.DataFrame, league: str, season_start_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise one raw CSV into (matches, odds), correctly aligned.

    This is the function callers should use. It guarantees the odds rows refer
    only to matches that actually made it into the matches frame.
    """
    matches = normalize_matches(raw, league, season_start_year)
    odds = extract_odds(raw, matches)
    return matches.drop(columns=["source_row"]), odds

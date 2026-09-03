"""Default DB-backed settings and seed lists. See docs/11-config.md and docs/02-data-sources.md.

These are *defaults*: on first boot they land in the `settings` table, after which the
DB wins and everything here is editable from the UI.
"""

from __future__ import annotations

# --- 2026/27 ruleset (README). Hard-coded because the optimiser depends on it. -----
SQUAD_SIZE = 15
POSITION_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
START_BUDGET = 1000  # tenths of a million
MAX_FREE_TRANSFERS = 5
HIT_COST = 4
CHIP_SET_1_LAST_GW = 19  # set 1 must be played before the GW19 deadline (2 Jan 2027)
CHIPS = ("wildcard", "freehit", "bboost", "3xc")
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
DEFCON_POINTS = 2

POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# FPL scores exactly four positions, but historical feeds do not all agree on that: the
# 2024-25 archive labels 20 players `AM`, which is enough to raise KeyError out of the
# simulator's scoring dicts and abort a whole gameweek — every 2024-25 backtest gameweek
# died this way. Anything unrecognised scores as a midfielder, which is what FPL does with
# an attacking midfielder anyway.
POSITION_ALIASES = {"AM": "MID", "DM": "MID", "CM": "MID", "WB": "DEF", "CB": "DEF",
                    "LB": "DEF", "RB": "DEF", "ST": "FWD", "CF": "FWD", "GKP": "GK"}


def normalise_position(position: str | None) -> str:
    """One of GK/DEF/MID/FWD, whatever the source called it."""
    p = (position or "").strip().upper()
    if p in POSITION_BY_ELEMENT_TYPE.values():
        return p
    return POSITION_ALIASES.get(p, "MID")

# --- Seed source registry (docs/02-data-sources.md) -------------------------------
# (id, display_name, category, requires_key, enabled_by_default, base_url, rate_limit/min)
SEED_SOURCES: list[tuple] = [
    ("fpl_official", "FPL Official API", "fpl", 0, 1, "https://fantasy.premierleague.com/api", 60),
    ("fpl_write", "FPL Write (unofficial)", "fpl", 1, 0, "https://users.premierleague.com", 6),
    ("vaastav_history", "vaastav FPL history", "fpl", 0, 1, "https://raw.githubusercontent.com", 60),
    ("livefpl", "LiveFPL ownership", "meta", 0, 1, "https://www.livefpl.net", 10),
    ("understat", "Understat", "stats", 0, 1, "https://understat.com", 20),
    ("fbref", "FBref (soccerdata)", "stats", 0, 1, "https://fbref.com", 15),
    ("sofascore", "SofaScore", "stats", 0, 1, "https://api.sofascore.com", 30),
    ("whoscored", "WhoScored", "stats", 0, 0, "https://www.whoscored.com", 6),
    ("transfermarkt", "Transfermarkt", "stats", 0, 1, "https://www.transfermarkt.co.uk", 10),
    ("football_data_org", "football-data.org", "stats", 1, 1, "https://api.football-data.org/v4", 10),
    ("api_football", "API-Football", "stats", 1, 1, "https://v3.football.api-sports.io", 30),
    ("sportmonks", "Sportmonks", "stats", 1, 1, "https://api.sportmonks.com/v3", 30),
    ("odds_api", "The Odds API", "odds", 1, 1, "https://api.the-odds-api.com/v4", 5),
    ("betfair", "Betfair Exchange", "odds", 1, 1, "https://api.betfair.com/exchange", 20),
    ("premier_injuries", "PremierInjuries", "injury", 0, 1, "https://www.premierinjuries.com", 10),
    ("physioroom", "PhysioRoom", "injury", 0, 1, "https://www.physioroom.com", 10),
    ("rss_news", "News RSS", "news", 0, 1, None, 120),
    ("setpieces", "Set-piece takers", "meta", 0, 1, None, 5),
    ("euro_fixtures", "European fixtures", "meta", 0, 1, None, 10),
    ("weather", "Open-Meteo", "meta", 0, 1, "https://api.open-meteo.com/v1", 60),
    ("referees", "Referee appointments", "meta", 0, 1, None, 5),
    ("youtube", "YouTube Data API", "video", 1, 1, "https://www.googleapis.com/youtube/v3", 60),
    ("reddit", "Reddit", "social", 1, 1, "https://oauth.reddit.com", 30),
    ("bluesky", "Bluesky", "social", 0, 1, "https://public.api.bsky.app", 60),
    ("twitter_scrape", "X (unofficial)", "social", 0, 1, None, 10),
]

# --- YouTube starter channels (docs/02-data-sources.md tier 5) ---------------------
# channel_id is resolved on first run via search; the title is the stable handle here.
SEED_YOUTUBE_CHANNELS = [
    "Let's Talk FPL",
    "FPL Mate",
    "FPL Harry",
    "FPL Raptor",
    "Above Average FPL",
    "Focal Fantasy",
    "FPL Family",
    "Planet FPL",
    "FPL BlackBox",
    "Fantasy Football Scout",
    "FPL Andy",
    "Elite FPL",
    "FPL Tips",
    "Fantasy Football Hub",
    "FPL Kiwi",
    "FPL Sonaldo",
    "Let's Talk Transfers",
    "The FPL Show",
]

SEED_RSS_FEEDS = [
    "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "https://www.skysports.com/rss/12040",
    "https://www.theguardian.com/football/rss",
    "https://www.theguardian.com/football/premierleague/rss",
    "https://talksport.com/football/feed/",
    "https://www.football365.com/feed",
    "https://www.90min.com/posts.rss",
    "https://metro.co.uk/sport/football/feed/",
    "https://www.mirror.co.uk/sport/football/?service=rss",
    "https://www.express.co.uk/posts/rss/65/football",
    "https://www.fantasyfootballscout.co.uk/feed/",
]

# Club RSS (BBC per-club feeds carry the actual team news).
SEED_CLUB_RSS = [
    f"https://feeds.bbci.co.uk/sport/football/teams/{slug}/rss.xml"
    for slug in [
        "arsenal", "aston-villa", "bournemouth", "brentford", "brighton-and-hove-albion",
        "burnley", "chelsea", "coventry-city", "crystal-palace", "everton", "fulham",
        "hull-city", "ipswich-town", "leeds-united", "liverpool", "manchester-city",
        "manchester-united", "newcastle-united", "nottingham-forest", "sunderland",
        "tottenham-hotspur", "west-ham-united", "wolverhampton-wanderers",
    ]
]

SEED_X_HANDLES = [
    "OptaJoe", "FPLStatistics", "FPL_Salah", "FantasyPL", "David_Ornstein",
    "FabrizioRomano", "PhysioScout", "BenDinnery", "OfficialFPL",
]

SEED_BLUESKY_HANDLES = ["bbcsport.bsky.social", "theathletic.bsky.social"]

SEED_SUBREDDITS = ["FantasyPL", "soccer"]

# Stadium coordinates for the weather connector (docs/02 tier 4).
STADIUM_COORDS: dict[str, tuple[str, float, float]] = {
    "Arsenal": ("Emirates Stadium", 51.5549, -0.1084),
    "Aston Villa": ("Villa Park", 52.5092, -1.8848),
    "Bournemouth": ("Vitality Stadium", 50.7348, -1.8391),
    "Brentford": ("Gtech Community Stadium", 51.4907, -0.2887),
    "Brighton": ("Amex Stadium", 50.8616, -0.0836),
    "Burnley": ("Turf Moor", 53.7890, -2.2303),
    "Chelsea": ("Stamford Bridge", 51.4817, -0.1910),
    "Coventry City": ("Coventry Building Society Arena", 52.4480, -1.4954),
    "Crystal Palace": ("Selhurst Park", 51.3983, -0.0855),
    "Everton": ("Hill Dickinson Stadium", 53.4013, -2.9967),
    "Fulham": ("Craven Cottage", 51.4749, -0.2217),
    "Hull City": ("MKM Stadium", 53.7460, -0.3679),
    "Ipswich Town": ("Portman Road", 52.0550, 1.1449),
    "Leeds": ("Elland Road", 53.7778, -1.5722),
    "Liverpool": ("Anfield", 53.4308, -2.9608),
    "Man City": ("Etihad Stadium", 53.4831, -2.2004),
    "Man Utd": ("Old Trafford", 53.4631, -2.2913),
    "Newcastle": ("St James' Park", 54.9756, -1.6217),
    "Nott'm Forest": ("City Ground", 52.9400, -1.1329),
    "Sunderland": ("Stadium of Light", 54.9145, -1.3882),
    "Spurs": ("Tottenham Hotspur Stadium", 51.6043, -0.0665),
    "West Ham": ("London Stadium", 51.5387, -0.0166),
    "Wolves": ("Molineux", 52.5903, -2.1303),
}

# Team nickname -> canonical FPL short name. Used by entity resolution.
TEAM_ALIASES: dict[str, str] = {
    "arsenal": "Arsenal", "gunners": "Arsenal", "afc": "Arsenal",
    "aston villa": "Aston Villa", "villa": "Aston Villa", "avfc": "Aston Villa",
    "bournemouth": "Bournemouth", "afc bournemouth": "Bournemouth", "cherries": "Bournemouth",
    "brentford": "Brentford", "bees": "Brentford",
    "brighton": "Brighton", "brighton and hove albion": "Brighton", "seagulls": "Brighton",
    "burnley": "Burnley", "clarets": "Burnley",
    "chelsea": "Chelsea", "blues": "Chelsea", "cfc": "Chelsea",
    "coventry": "Coventry City", "coventry city": "Coventry City",
    "sky blues": "Coventry City",
    "crystal palace": "Crystal Palace", "palace": "Crystal Palace", "eagles": "Crystal Palace",
    "everton": "Everton", "toffees": "Everton",
    "fulham": "Fulham", "cottagers": "Fulham",
    "hull": "Hull City", "hull city": "Hull City", "tigers": "Hull City",
    "ipswich": "Ipswich Town", "ipswich town": "Ipswich Town",
    "tractor boys": "Ipswich Town",
    "leeds": "Leeds", "leeds united": "Leeds",
    "liverpool": "Liverpool", "lfc": "Liverpool", "reds": "Liverpool",
    "man city": "Man City", "manchester city": "Man City", "city": "Man City", "mcfc": "Man City",
    "man utd": "Man Utd", "manchester united": "Man Utd", "man united": "Man Utd",
    "united": "Man Utd", "mufc": "Man Utd", "red devils": "Man Utd",
    "newcastle": "Newcastle", "newcastle united": "Newcastle", "magpies": "Newcastle",
    "nottingham forest": "Nott'm Forest", "forest": "Nott'm Forest", "nffc": "Nott'm Forest",
    "nott'm forest": "Nott'm Forest",
    "sunderland": "Sunderland", "black cats": "Sunderland",
    "spurs": "Spurs", "tottenham": "Spurs", "tottenham hotspur": "Spurs", "thfc": "Spurs",
    "west ham": "West Ham", "west ham united": "West Ham", "hammers": "West Ham",
    "wolves": "Wolves", "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves",
}

# Common player nicknames that fuzzy matching will never get on its own.
SEED_PLAYER_ALIASES: dict[str, str] = {
    "taa": "Trent Alexander-Arnold",
    "trent": "Trent Alexander-Arnold",
    "kdb": "Kevin De Bruyne",
    "vvd": "Virgil van Dijk",
    "bruno": "Bruno Fernandes",
    "big dog": "Erling Haaland",
    "sonny": "Son Heung-min",
    "cunha": "Matheus Cunha",
    "mitoma": "Kaoru Mitoma",
    "gabby jesus": "Gabriel Jesus",
    "the ginger pep": "Kieran McKenna",
}

DEFAULT_GLOBAL_SETTINGS: dict = {
    "llm.tasks": {
        "extract_claims": {"model": "", "temperature": 0.1},
        "resolve_entity": {"model": "", "temperature": 0.0},
        "classify_injury_severity": {"model": "", "temperature": 0.1},
        "dedupe_claims": {"model": "", "temperature": 0.0},
        "summarise_video": {"model": "", "temperature": 0.3},
        "summarise_player_week": {"model": "", "temperature": 0.3},
        "explain_recommendation": {"model": "", "temperature": 0.4},
        "critique_recommendation": {"model": "", "temperature": 0.5},
        "chat": {"model": "", "temperature": 0.6},
        "weekly_digest": {"model": "", "temperature": 0.5},
        "settings_assistant": {"model": "", "temperature": 0.2},
    },
    "sources.enabled": {s[0]: bool(s[4]) for s in SEED_SOURCES},
    "sources.cadence": {
        "fpl_bootstrap": "0 * * * *",
        "fpl_fixtures": "5 * * * *",
        "fpl_element_summaries": "30 4 * * *",
        "fpl_entry_sync": "0 */6 * * *",
        "odds_poll": "0 */4 * * *",
        "injury_scrape": "15 */3 * * *",
        "lineups_poll": "20 * * * *",
        "news_rss": "*/20 * * * *",
        "youtube_tracked": "40 */3 * * *",
        "youtube_discovery": "0 5 * * *",
        "transcripts": "*/15 * * * *",
        "social_x": "*/30 * * * *",
        "bluesky": "*/30 * * * *",
        "reddit": "*/30 * * * *",
        "understat": "0 3 * * 2",
        "fbref": "0 4 * * 2",
        "sofascore_ratings": "0 5 * * 2",
        "transfermarkt": "0 6 * * 3",
        "weather": "0 7 * * *",
        "extract_claims": "*/10 * * * *",
        "build_features": "*/30 * * * *",
        "predict": "0 6,12,20 * * *",
        "optimise_all_squads": "20 6,12,20 * * *",
        "train_models": "0 10 * * 2",
        "resolve_pundit_calls": "0 11 * * 2",
        "discord_digest": "0 8 * * *",
        "vacuum_analyze": "0 4 1 * *",
    },
    "youtube.channels": [{"title": t, "channel_id": "", "tracked": True, "trust_weight": 1.0}
                         for t in SEED_YOUTUBE_CHANNELS],
    "youtube.discovery_queries": [
        "FPL GW{gw} team selection",
        "FPL gameweek {gw} transfer tips",
        "fantasy premier league gameweek {gw} captain",
        "FPL {gw} differentials",
    ],
    "youtube.discovery_min_views": 2000,
    "rss.feeds": SEED_RSS_FEEDS + SEED_CLUB_RSS,
    "rss.google_news_watchlist_query": True,
    "x.handles": SEED_X_HANDLES,
    "bluesky.handles": SEED_BLUESKY_HANDLES,
    "reddit.subreddits": SEED_SUBREDDITS,
    "text.trust_weights": {
        "tier1_journalist": 1.5,
        "official_club": 2.0,
        "youtube_default": 0.8,
        "reddit": 0.4,
        "rss_default": 1.0,
    },
    "adjustment.enabled": True,
    "adjustment.max_points": 2.0,
    "adjustment.max_fraction": 0.25,
    "model.odds_blend_weight": 0.65,
    "model.bps_regime_prior_fixtures": 40,
    "ui.theme": "dark",
    "watchlist": [],
}

DEFAULT_SQUAD_SETTINGS: dict = {
    "risk": 0.0,
    "horizon_gws": 5,
    "horizon_decay": 0.84,
    "bench_weight": 0.12,
    "max_hits_per_gw": 1,
    "min_expected_gain_to_act": 0.8,
    "prefer_differentials": False,
    "rank_mode": "maximise_points",
    "leagues": [],
    "banned_clubs": [],
    "locked_players": [],
    "must_own": [],
    "chip_strategy": {
        "wildcard_earliest_gw": 4,
        "bench_boost_prefer_dgw": True,
        "save_second_set": True,
    },
    "price_bonus_weight": 0.0,
    "auto_sync_from_fpl": True,
    "max_transfers_per_gw": 3,
    # Cap on positions 12-15. None derives it from the cheapest legal bench.
    "max_bench_value": None,
    "notes": "",
}

SQUAD_COLOURS = ["#4ADE80", "#60A5FA", "#F472B6", "#FBBF24", "#A78BFA", "#22D3EE", "#FB923C"]

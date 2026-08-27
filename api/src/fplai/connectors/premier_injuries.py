"""premierinjuries.com — the best structured UK injury table. docs/02 tier 4.

FLAKY by nature: HTML changes. The parser is defensive and a zero-row parse is
surfaced as a source-health alert rather than silently succeeding.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from ..resolve.entities import resolve_name
from .base import Connector, IngestContext, ParsedBatch, RawDoc, fetch_url, utcnow

log = logging.getLogger(__name__)

URL = "https://www.premierinjuries.com/injury-table.php"

STATUS_MAP = {
    "doubtful": "doubt", "doubt": "doubt", "50%": "doubt", "75%": "doubt", "25%": "doubt",
    "out": "injured", "injured": "injured", "unavailable": "injured",
    "suspended": "suspended", "available": "available", "fit": "available",
}


def _text(node) -> str:
    return re.sub(r"\s+", " ", node.text(strip=True) if node else "").strip()


def map_status(raw: str) -> tuple[str, int | None]:
    low = raw.lower()
    pct = None
    m = re.search(r"(\d{1,3})\s*%", low)
    if m:
        pct = int(m.group(1))
    for key, val in STATUS_MAP.items():
        if key in low:
            return val, pct
    return ("doubt" if pct is not None and pct < 100 else "unknown"), pct


class PremierInjuriesConnector(Connector):
    id = "premier_injuries"
    category = "injury"
    default_cadence = "15 */3 * * *"
    rate_limit_per_min = 10
    scrapey = True
    parser_version = 1

    async def fetch(self, ctx: IngestContext) -> AsyncIterator[RawDoc]:
        r = await fetch_url(URL, self.id, per_min=self.rate_limit_per_min)
        yield RawDoc("injury_table", r.text, external_id="table", url=URL)

    def parse(self, doc: RawDoc) -> ParsedBatch:
        from selectolax.parser import HTMLParser

        tree = HTMLParser(doc.payload)
        observed = utcnow()
        parsed: list[dict] = []
        for row in tree.css("table tr"):
            cells = [_text(c) for c in row.css("td")]
            if len(cells) < 4:
                continue
            name, club, issue, status = cells[0], cells[1], cells[2], cells[3]
            expected = cells[4] if len(cells) > 4 else None
            if not name or name.lower() in ("player", "name"):
                continue
            parsed.append(
                {"name": name, "club": club, "issue": issue, "status": status,
                 "expected": expected}
            )

        if not parsed:
            log.warning("premier_injuries parsed zero rows — HTML layout probably changed")

        b = ParsedBatch()

        def _rows(conn) -> int:
            from ..config import get_settings

            season = get_settings().current_season
            n = 0
            for p in parsed:
                res = resolve_name(p["name"], p["club"], season)
                if not res.player_id:
                    continue
                status, pct = map_status(p["status"])
                conn.execute(
                    "INSERT OR IGNORE INTO availability(player_id,source_id,observed_at,status,"
                    "chance_pct,issue,expected_return,note) "
                    "VALUES(?,'premier_injuries',?,?,?,?,?,?)",
                    (res.player_id, observed, status, pct, p["issue"][:120],
                     (p["expected"] or "")[:60], ""),
                )
                n += 1
            return n

        b.defer(_rows)
        return b

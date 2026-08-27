"""PhysioRoom — second injury opinion. docs/02 tier 4.

Its value is disagreement: two sources differing on a return date is itself a signal,
consumed by the `source_disagreement_score` feature.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from ..resolve.entities import resolve_name
from .base import Connector, IngestContext, ParsedBatch, RawDoc, fetch_url, utcnow
from .premier_injuries import map_status

log = logging.getLogger(__name__)

URL = "https://www.physioroom.com/injury-table/premier-league/"


class PhysioRoomConnector(Connector):
    id = "physioroom"
    category = "injury"
    default_cadence = "45 */3 * * *"
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
        rows = []
        current_club = None
        for node in tree.css("h2, h3, table tr"):
            if node.tag in ("h2", "h3"):
                current_club = re.sub(r"\s+", " ", node.text(strip=True)).strip()
                continue
            cells = [re.sub(r"\s+", " ", c.text(strip=True)) for c in node.css("td")]
            if len(cells) >= 3 and cells[0] and cells[0].lower() != "player":
                rows.append(
                    {
                        "name": cells[0],
                        "club": current_club,
                        "issue": cells[1],
                        "status": cells[2],
                        "expected": cells[3] if len(cells) > 3 else None,
                    }
                )
        if not rows:
            log.warning("physioroom parsed zero rows — layout probably changed")

        b = ParsedBatch()

        def _rows(conn) -> int:
            from ..config import get_settings

            season = get_settings().current_season
            n = 0
            for p in rows:
                res = resolve_name(p["name"], p["club"], season)
                if not res.player_id:
                    continue
                status, pct = map_status(p["status"])
                conn.execute(
                    "INSERT OR IGNORE INTO availability(player_id,source_id,observed_at,status,"
                    "chance_pct,issue,expected_return,note) VALUES(?,'physioroom',?,?,?,?,?,'')",
                    (res.player_id, observed, status, pct, p["issue"][:120],
                     (p["expected"] or "")[:60]),
                )
                n += 1
            return n

        b.defer(_rows)
        return b

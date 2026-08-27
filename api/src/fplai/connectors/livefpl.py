"""LiveFPL — effective ownership and top-10k ownership. docs/02 tier 1, HIGH.

EO is what makes the risk/differential setting mean anything. FLAKY: a zero-row parse
is logged loudly so the Sources screen shows it, rather than silently writing nothing.
"""

from __future__ import annotations

import logging
import re

from ..resolve.entities import resolve_name
from .base import Connector, ParsedBatch, RawDoc, fetch_url, utcnow
from .fpl_official import current_gameweek

log = logging.getLogger(__name__)

# The old /effective_ownership path now 302s to a broken double-slash URL that 404s.
# The live page is /EO on plan.livefpl.net, and it 500s until a gameweek has been played
# — EO is undefined before anyone has captained anyone, so a pre-season failure is normal.
URL = "https://plan.livefpl.net/EO"


class LiveFplConnector(Connector):
    id = "livefpl"
    category = "meta"
    default_cadence = "0 */6 * * *"
    rate_limit_per_min = 10
    scrapey = True
    parser_version = 1

    async def fetch(self, ctx):
        r = await fetch_url(URL, self.id, per_min=self.rate_limit_per_min)
        yield RawDoc(
            "ownership", r.text, external_id=f"eo:{current_gameweek(ctx.season_id)}", url=URL
        )

    def parse(self, doc: RawDoc) -> ParsedBatch:
        from selectolax.parser import HTMLParser

        tree = HTMLParser(doc.payload)
        gw = int(doc.external_id.split(":")[1])
        observed = utcnow()
        rows = []
        for tr in tree.css("table tr"):
            cells = [re.sub(r"\s+", " ", c.text(strip=True)) for c in tr.css("td")]
            if len(cells) < 3 or not cells[0] or cells[0].lower() == "player":
                continue
            rows.append(cells)
        if not rows:
            log.warning("livefpl parsed zero ownership rows — HTML layout probably changed")

        b = ParsedBatch()

        def _rows(conn) -> int:
            from ..config import get_settings

            season = get_settings().current_season
            n = 0
            for cells in rows:
                res = resolve_name(cells[0], None, season)
                if not res.player_id:
                    continue
                owned, eo = _pct(cells[1]), _pct(cells[2])
                cap = _pct(cells[3]) if len(cells) > 3 else None
                for scope, present in (("top10k", eo), ("overall", owned)):
                    if present is None:
                        continue
                    conn.execute(
                        "INSERT OR REPLACE INTO ownership_snapshots(player_id,gameweek,season_id,"
                        "scope,owned_pct,captained_pct,effective_ownership,observed_at) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (res.player_id, gw, season, scope, owned, cap, eo, observed),
                    )
                    n += 1
            return n

        b.defer(_rows)
        return b


def _pct(s: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", s or "")
    return float(m.group(1)) if m else None

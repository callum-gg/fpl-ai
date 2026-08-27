"""Set-piece takers: the community sheet exported as CSV. docs/02 tier 4.

Penalty duty is worth roughly half a point per game to a forward; corner and free-kick
duty drives assists. Cheap data, real effect.
"""

from __future__ import annotations

import csv
import io

from ..db.settings_store import global_settings
from ..resolve.entities import resolve_name
from .base import Connector, ParsedBatch, RawDoc, fetch_url, utcnow

ROLES = {
    "penalt": "penalties",
    "direct free": "direct_fk",
    "corners left": "corners_left",
    "corners right": "corners_right",
    "corner": "corners_left",
}


class SetPiecesConnector(Connector):
    id = "setpieces"
    category = "meta"
    default_cadence = "0 6 * * 3"
    rate_limit_per_min = 5
    parser_version = 1

    async def fetch(self, ctx):
        url = global_settings().get("setpieces.csv_url") or ctx.params.get("csv_url")
        if not url:
            return
        r = await fetch_url(url, self.id, per_min=self.rate_limit_per_min)
        yield RawDoc("setpiece_sheet", r.text, external_id=ctx.season_id, url=url)

    def parse(self, doc: RawDoc) -> ParsedBatch:
        rows = list(csv.DictReader(io.StringIO(doc.payload)))
        season = doc.external_id
        observed = utcnow()
        b = ParsedBatch()

        def _roles(conn) -> int:
            n = 0
            for r in rows:
                club = r.get("Team") or r.get("team")
                for col, value in r.items():
                    if not value or not col:
                        continue
                    role = next((v for k, v in ROLES.items() if k in col.lower()), None)
                    if role is None:
                        continue
                    names = [p.strip() for p in str(value).split(",") if p.strip()]
                    for rank, name in enumerate(names, start=1):
                        res = resolve_name(name, club, season)
                        if not res.player_id:
                            continue
                        conn.execute(
                            "INSERT OR REPLACE INTO set_piece_roles(player_id,season_id,role,rank,"
                            "source_id,observed_at) VALUES(?,?,?,?,'setpieces',?)",
                            (res.player_id, season, role, rank, observed),
                        )
                        n += 1
            return n

        b.defer(_roles)
        return b

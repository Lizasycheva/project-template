import os
import json
import time
import pathlib
import requests
from typing import Iterable, Dict, Any, List, Optional

DEFAULT_DATA_DIR = os.environ.get("NHL_RAW_DIR", os.path.join(pathlib.Path(__file__).parent, "raw"))
SLEEP_BETWEEN_CALLS = float(os.environ.get("NHL_API_SLEEP", "0.2"))

class NHLPBPDownloader:
    """
    Download & cache NHL play-by-play JSON from the official stats API.

    - Discovers games via the schedule endpoint (no hard-coded IDs/counts).
    - Saves one file per game: {gamePk}.json
    - Supports regular season ('R') and playoffs ('P').

    Example:
        dl = NHLPBPDownloader()
        dl.download_season("20162017", game_types=("R","P"))
    """
    BASE = "https://statsapi.web.nhl.com/api/v1"
    UA = {"User-Agent": "ift6758-m1-downloader/1.0 (+https://example.com)"}

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(self.UA)

    # ---------- Public API ----------

    def download_season(self, season: str, game_types: Iterable[str] = ("R","P")) -> None:
        """season: 'YYYYYYYY' (e.g., '20162017')."""
        self._assert_season(season)
        game_pks = self._list_games_for_season(season, game_types)
        for pk in game_pks:
            self._download_game_pbp(pk)

    def download_many_seasons(self, seasons: Iterable[str], game_types: Iterable[str] = ("R","P")) -> None:
        for s in seasons:
            self.download_season(s, game_types)

    # ---------- Internals ----------

    def _list_games_for_season(self, season: str, game_types: Iterable[str]) -> List[int]:
        # Use schedule endpoint to discover actual games
        url = f"{self.BASE}/schedule?season={season}"
        data = self._get_json(url)
        want = set(game_types)
        pks: List[int] = []
        for day in data.get("dates", []):
            for g in day.get("games", []):
                gt = g.get("gameType")  # 'R', 'P', etc.
                if gt in want:
                    pk = g.get("gamePk")
                    if isinstance(pk, int):
                        pks.append(pk)
        return pks

    def _download_game_pbp(self, game_pk: int) -> None:
        out = os.path.join(self.data_dir, f"{game_pk}.json")
        if os.path.exists(out):
            return  # cached

        url = f"{self.BASE}/game/{game_pk}/feed/live"
        data = self._get_json(url)
        # Basic sanity: ensure this looks like a game feed
        if not data or "gameData" not in data:
            return  # skip silently or log

        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f)
        time.sleep(SLEEP_BETWEEN_CALLS)

    def _get_json(self, url: str, retries: int = 3, timeout: float = 20.0) -> Optional[Dict[str, Any]]:
        for i in range(retries):
            try:
                r = self.session.get(url, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                # 404 on schedule days off is possible; treat as empty
                if r.status_code == 404:
                    return {}
            except requests.RequestException:
                pass
            time.sleep(0.5 * (i + 1))
        return {}

    @staticmethod
    def _assert_season(season: str) -> None:
        # Expect 'YYYYYYYY' and inclusive range 2016-17 .. 2023-24
        assert isinstance(season, str) and len(season) == 8 and season.isdigit()
        start = int(season[:4]); end = int(season[4:])
        assert end == start + 1
        assert 2016 <= start <= 2023

if __name__ == "__main__":
    seasons = [f"{y}{y+1}" for y in range(2016, 2024)]  # 20162017..20232024
    NHLPBPDownloader().download_many_seasons(seasons, game_types=("R","P"))

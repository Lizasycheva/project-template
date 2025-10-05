from .ApiClient import ApiClient
from .FileSystemCache import FileSystemCache

class NHLDataLoader:
    def __init__(self, cache=None):
        self.api = ApiClient()
        self.cache = cache or FileSystemCache()

    def fetch_game(self, game_id: str) -> dict:
        cached = self.cache.get(game_id)
        if cached: return cached
        data = self.api.get_feed_live(game_id)
        self.cache.set(game_id, data)
        self.cache.dump(game_id, data)
        return data

    def fetch_season(self, season_start_year: int, game_type: str, max_games=1320):
        for n in range(1, max_games+1):
            gid = f"{season_start_year}{game_type}{n:04d}"
            try:
                self.fetch_game(gid)
            except Exception:
                # s'il n'y a plus de matchs valides, on “tombe dans le vide” et continue
                continue

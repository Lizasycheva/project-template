import os, json

class FileSystemCache:
    def __init__(self, cache_dir="ift6758/data/storage/cache",
                 dump_dir="ift6758/data/storage/dump"):
        self.cache_dir, self.dump_dir = cache_dir, dump_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(dump_dir, exist_ok=True)

    def get(self, key: str):
        p = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
        return None

    def set(self, key: str, data: dict):
        p = os.path.join(self.cache_dir, f"{key}.json")
        with open(p, "w") as f: json.dump(data, f)

    def dump(self, key: str, data: dict):
        p = os.path.join(self.dump_dir, f"{key}.json")
        with open(p, "w") as f: json.dump(data, f)

from pathlib import Path
import json
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Dropdown, ToggleButtons, VBox, HBox
from ift6758.data.data_acquisition import NHLDataLoader

# Chemins
DUMP_DIR  = Path("IFT6758-Projet/data/storage/dump")
RINK_IMG  = Path("IFT6758-Projet/figures/nfl_rink.png") 
loader    = NHLDataLoader()

def make_game_id(season_start_year: int, game_type: str, game_number: int) -> str:
    return f"{season_start_year}{game_type}{game_number:04d}"

def load_local_game_json(game_id: str):
    p = DUMP_DIR / f"{game_id}.json"
    if p.exists():
        with open(p) as f: 
            return json.load(f)
    return None

def get_events(feed: dict):
    try:
        return feed["liveData"]["plays"]["allPlays"]
    except Exception:
        return []

def parse_event(evt: dict):
    r = {}
    r["eventType"]  = evt.get("result", {}).get("eventTypeId")
    r["description"]= evt.get("result", {}).get("description")
    r["period"]     = evt.get("about", {}).get("period")
    r["periodTime"] = evt.get("about", {}).get("periodTime")
    r["x"]          = evt.get("coordinates", {}).get("x")
    r["y"]          = evt.get("coordinates", {}).get("y")
    r["team"]       = (evt.get("team", {}) or {}).get("name")
    r["shotType"]   = evt.get("result", {}).get("secondaryType")
    r["isGoal"]     = (r["eventType"] == "GOAL")
    return r

def show_event_info(info: dict):
    lines = [
        f"[{info.get('period','?')}] {info.get('periodTime','??:??')}",
        f"{info.get('eventType','?')}: {info.get('description','')}",
        f"Équipe: {info.get('team','?')}"
    ]
    if info.get("shotType"): lines.append(f"Type de tir: {info['shotType']}")
    if info.get("isGoal"):   lines.append("BUT !")
    print("\n".join(lines))

# Patinoire dessin si le fichier ne fonctionne pas -> à effacer?
def plot_on_rink(x, y):
    fig = plt.figure(figsize=(8, 4))
    ax  = plt.gca()
    try:
        img = plt.imread(RINK_IMG)
        ax.imshow(img)
        h, w = img.shape[0], img.shape[1]
    except Exception:
        h, w = 400, 800
        ax.imshow([[1,1],[1,1]])
        ax.set_xlim(0, w); ax.set_ylim(h, 0)

    ax.axis("off")
    if x is not None and y is not None:
        # Conversion coords NHL
        px = ((x + 100.0) / 200.0) * w
        py = (1.0 - ((y + 42.5) / 85.0)) * h
        ax.scatter([px], [py], s=80)
        ax.set_title(f"(x={x}, y={y})")
    else:
        ax.set_title("Pas de coordonnées pour cet événement.")
    plt.show()

# widgets
seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
w_season = Dropdown(options=seasons, value=2017, description="Saison")
w_type   = ToggleButtons(options=[("Régulière","02"), ("Séries","03")], value="02", description="Type")
w_num    = IntSlider(value=1, min=1, max=1320, step=1, description="Match #", continuous_update=False)
w_idx    = IntSlider(value=0, min=0, max=50, step=1, description="Événement", continuous_update=False)

def _load_game(season, gtype, gnum):
    gid  = make_game_id(season, gtype, gnum)
    data = load_local_game_json(gid)
    if data is None:
        try:
            data = loader.fetch_game(gid)
        except Exception:
            return gid, []
    return gid, get_events(data)

def _update(season, gtype, gnum, idx):
    gid, evts = _load_game(season, gtype, gnum)
    print(f"GAME_ID = {gid} | #événements = {len(evts)}")
    if not evts:
        print("Aucun événement disponible.")
        return
    w_idx.max = max(0, len(evts) - 1)
    evt = parse_event(evts[min(idx, len(evts)-1)])
    show_event_info(evt)
    plot_on_rink(evt["x"], evt["y"])

# Affichage
ui = VBox([HBox([w_season, w_type, w_num]), w_idx])
display(ui)
interact(_update, season=w_season, gtype=w_type, gnum=w_num, idx=w_idx);

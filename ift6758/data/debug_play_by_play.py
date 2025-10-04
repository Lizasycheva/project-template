#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports de base
from ift6758.data.data_acquisition import NHLDataLoader
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ipywidgets import interact, fixed, IntSlider, Dropdown, ToggleButtons, VBox, HBox

# Vérifier le dossier courant (doit être la racine du repo pour que l'import ift6758 fonctionne)
import os; print('cwd =', os.getcwd())


# In[ ]:


# Chemins et utilitaires
DUMP_DIR = Path('ift6758/data/storage/dump')
ASSETS_DIR = Path('assets')
RINK_IMG = ASSETS_DIR / 'rink.png'
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def make_game_id(season_start_year: int, game_type: str, game_number: int) -> str:
    """Construit GAME_ID = YYYY + game_type (02 ou 03) + ####"""
    return f"{season_start_year}{game_type}{game_number:04d}"

def load_local_game_json(game_id: str):
    path = DUMP_DIR / f"{game_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# In[ ]:


# Création d'une image de patinoire "placeholder" si assets/rink.png est absent.
# Pour rester simple, on crée une image blanche sur laquelle on placera le point.
import numpy as np
if not RINK_IMG.exists():
    h, w = 400, 800
    img = np.ones((h, w, 3), dtype=np.float32)  # blanc
    # Sauvegarde via matplotlib (aucun style/couleur spécifique défini)
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(RINK_IMG, bbox_inches='tight', pad_inches=0)
    plt.close()
RINK_IMG


# In[ ]:


# Fonctions d'accès et de parsing des événements
def get_events(data: dict):
    try:
        return data["liveData"]["plays"]["allPlays"]
    except Exception:
        return []

def parse_event(evt: dict):
    res = {}
    res["eventType"] = evt.get("result", {}).get("eventTypeId")
    res["description"] = evt.get("result", {}).get("description")
    about = evt.get("about", {})
    res["period"] = about.get("period")
    res["periodTime"] = about.get("periodTime")
    coords = evt.get("coordinates", {})
    res["x"] = coords.get("x", None)
    res["y"] = coords.get("y", None)
    team = evt.get("team", {})
    res["team"] = team.get("name") if isinstance(team, dict) else None
    res["shotType"] = evt.get("result", {}).get("secondaryType")
    res["isGoal"] = (res["eventType"] == "GOAL")
    return res


# In[ ]:


# Affichage d'un événement (texte) et tracé du point sur la patinoire
def show_event_info(res: dict):
    lines = [
        f"[{res.get('period','?')}] {res.get('periodTime','??:??')}",
        f"{res.get('eventType','?')}: {res.get('description','')}",
        f"Équipe: {res.get('team','?')}"
    ]
    if res.get("shotType"): lines.append(f"Type de tir: {res['shotType']}")
    if res.get("isGoal"):   lines.append("BUT !")
    print("\n".join(lines))

def plot_on_rink(x, y):
    img = mpimg.imread(RINK_IMG)
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.axis('off')
    h, w = img.shape[0], img.shape[1]
    # Mapping simple des coords NHL (x ∈ [-100,100], y ∈ [-42.5,42.5]) aux pixels
    # Note: ceci est un mapping linéaire approximatif pour le débogage interactif
    if x is not None and y is not None:
        # Convertir en [0,1]
        xn = (x + 100.0) / 200.0
        yn = (y + 42.5) / 85.0
        px = xn * w
        # Les y pixels augmentent vers le bas, donc on inverse
        py = (1 - yn) * h
        plt.scatter([px], [py], s=80)
        plt.title(f"(x={x}, y={y})")
    else:
        plt.title("Pas de coordonnées pour cet événement")
    plt.show()


# In[ ]:


# Widgets interactifs
loader = NHLDataLoader()

seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
w_season = Dropdown(options=seasons, value=2017, description='Saison')
w_type   = ToggleButtons(options=[('Régulière','02'), ('Séries','03')],
                         value='02', description='Type')
w_num    = IntSlider(value=1, min=1, max=1320, step=1, description='Match #', continuous_update=False)
w_idx    = IntSlider(value=0, min=0, max=50, step=1, description='Événement', continuous_update=False)

def _load_game_data(season, gtype, gnum):
    gid = make_game_id(season, gtype, gnum)
    data = load_local_game_json(gid)
    if data is None:
        try:
            data = loader.fetch_game(gid)
        except Exception as e:
            print(f"Échec téléchargement {gid}: {e}")
            return gid, None, []
    evts = get_events(data)
    return gid, data, evts

def update(season, gtype, gnum, idx):
    gid, data, evts = _load_game_data(season, gtype, gnum)
    print(f"GAME_ID = {gid}  |  #événements = {len(evts)}")
    if not evts:
        print("Aucun événement disponible (match inexistant, pas encore joué, ou erreur réseau).")
        return
    w_idx.max = max(0, len(evts)-1)
    evt = parse_event(evts[min(idx, len(evts)-1)])
    show_event_info(evt)
    plot_on_rink(evt["x"], evt["y"]) 

ui = VBox([HBox([w_season, w_type, w_num]), w_idx])
out = interact(update, season=w_season, gtype=w_type, gnum=w_num, idx=w_idx)
ui


from pathlib import Path
import json
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Dropdown, ToggleButtons, VBox, HBox
from ift6758.data.data_acquisition import NHLDataLoader


# Initialisation des chemins 

DUMP_DIR  = Path("IFT6758-Projet/data/storage/dump")          
RINK_IMG  = Path("IFT6758-Projet/figures/nfl_rink.png")       
loader    = NHLDataLoader()                                   # Objet pour télécharger les données via l'API NHL

# Génère un identifiant de match unique (GAME_ID) selon le format LNH
def make_game_id(season_start_year: int, game_type: str, game_number: int) -> str:
    return f"{season_start_year}{game_type}{game_number:04d}"

# Charge le fichier JSON d’un match localement s’il existe
def load_local_game_json(game_id: str):
    p = DUMP_DIR / f"{game_id}.json"
    if p.exists():                        # Si le fichier est trouvé localement
        with open(p) as f: 
            return json.load(f)           # On charge et retourne le contenu JSON
    return None                           # Sinon, on retourne None

# Extrait la liste des événements du flux JSON du match
def get_events(feed: dict):
    try:
        return feed["liveData"]["plays"]["allPlays"]
    except Exception:
        return []                         # Retourne une liste vide si la clé n'existe pas

# Transforme un événement brut (JSON) en dictionnaire simplifié (type, description, coords, équipe, etc.)
def parse_event(evt: dict):
    r = {}
    r["eventType"]  = evt.get("result", {}).get("eventTypeId")        # Type d’événement (SHOT, GOAL, etc.)
    r["description"]= evt.get("result", {}).get("description")        # Texte descriptif
    r["period"]     = evt.get("about", {}).get("period")              # Période du match
    r["periodTime"] = evt.get("about", {}).get("periodTime")          # Temps dans la période
    r["x"]          = evt.get("coordinates", {}).get("x")             # Coordonnée X sur la glace
    r["y"]          = evt.get("coordinates", {}).get("y")             # Coordonnée Y sur la glace
    r["team"]       = (evt.get("team", {}) or {}).get("name")         # Nom de l’équipe concernée
    r["shotType"]   = evt.get("result", {}).get("secondaryType")      # Type de tir (slap shot, wrist shot, etc.)
    r["isGoal"]     = (r["eventType"] == "GOAL")                      # Booléen indiquant si c’est un but
    return r

# Affiche les informations d’un événement de façon lisible dans la console
def show_event_info(info: dict):
    lines = [
        f"[{info.get('period','?')}] {info.get('periodTime','??:??')}",     # Période et temps
        f"{info.get('eventType','?')}: {info.get('description','')}",       # Type + description
        f"Équipe: {info.get('team','?')}"                                   # Équipe
    ]
    if info.get("shotType"): lines.append(f"Type de tir: {info['shotType']}") # Type de tir (si dispo)
    if info.get("isGoal"):   lines.append("BUT !")                            # Indique un but
    print("\n".join(lines))


# Affiche la position de l’événement sur une image de patinoire
def plot_on_rink(x, y):
    fig = plt.figure(figsize=(8, 4))
    ax  = plt.gca()
    try:
        img = plt.imread(RINK_IMG)          # Charger l’image de patinoire
        ax.imshow(img)
        h, w = img.shape[0], img.shape[1]
    except Exception:
        # Si le fichier de la patinoire n’est pas trouvé, on affiche un fond vide
        h, w = 400, 800
        ax.imshow([[1,1],[1,1]])
        ax.set_xlim(0, w); ax.set_ylim(h, 0)

    ax.axis("off")
    if x is not None and y is not None:
        # Conversion des coordonnées NHL
        px = ((x + 100.0) / 200.0) * w
        py = (1.0 - ((y + 42.5) / 85.0)) * h
        ax.scatter([px], [py], s=80)        # Place un point sur la glace
        ax.set_title(f"(x={x}, y={y})")
    else:
        ax.set_title("Pas de coordonnées pour cet événement.")
    plt.show()


# Définition des widgets interactifs : saison, type de match, numéro, événement
seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
w_season = Dropdown(options=seasons, value=2017, description="Saison")              # Choix de saison
w_type   = ToggleButtons(options=[("Régulière","02"), ("Séries","03")], value="02", description="Type") # Type de match
w_num    = IntSlider(value=1, min=1, max=1320, step=1, description="Match #", continuous_update=False)  # Numéro du match
w_idx    = IntSlider(value=0, min=0, max=50, step=1, description="Événement", continuous_update=False)  # Événement


# Charger les événements pour un match donné
def _load_game(season, gtype, gnum):
    gid  = make_game_id(season, gtype, gnum)         # Création du GAME_ID
    data = load_local_game_json(gid)                 # Tente de charger localement
    if data is None:                                 # Si absent, tente de télécharger via API
        try:
            data = loader.fetch_game(gid)
        except Exception:
            return gid, []
    return gid, get_events(data)                     # Retourne la liste des événements

# Met à jour l’affichage lorsqu’un widget change (slider ou dropdown)
def _update(season, gtype, gnum, idx):
    gid, evts = _load_game(season, gtype, gnum)          # Charge les données du match
    print(f"GAME_ID = {gid} | #événements = {len(evts)}")# Affiche info match
    if not evts:
        print("Aucun événement disponible.")
        return
    w_idx.max = max(0, len(evts) - 1)                    # Ajuste le max du slider "événement"
    evt = parse_event(evts[min(idx, len(evts)-1)])        # Récupère l’événement courant
    show_event_info(evt)                                  # Affiche les infos textuelles
    plot_on_rink(evt["x"], evt["y"])                      # Affiche la position sur la glace

# Interface graphique (widgets)
ui = VBox([HBox([w_season, w_type, w_num]), w_idx])      
display(ui)                                               # Affiche le panneau d’interaction

interact(_update, season=w_season, gtype=w_type, gnum=w_num, idx=w_idx);

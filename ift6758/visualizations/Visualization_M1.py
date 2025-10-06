"""
Milestone 1 – Visualisations (Sections 4 & 5)

Assumptions about the cleaned dataframe (your ETL from Tasks 1–3 should output this):
  Required columns per row (SHOT or GOAL events only):
    - season (str, e.g., "2018-19")
    - gameId (int or str)
    - period (int, 1..5)
    - eventType (str, one of {"SHOT", "GOAL"})
    - team (str team name or tri-code)
    - shooter (str)  # optional
    - goalie (str)   # optional
    - shotType (str, e.g., "Wrist Shot", "Slap Shot", ...; can be None)
    - emptyNet (bool)
    - strength (str, e.g., "EVEN", "PP", "SH")  # can be None for shots
    - x (float), y (float)  # NHL play-by-play rink coords, center ice at (0,0)

  Notes on coordinates:
    • We normalise to an "attacking-right" convention so all shots go toward +x.
    • NHL goals are near x ≈ ±89 ft in the event feed. We treat the attacking net
      as (goal_x, 0) with goal_x = +89 after normalisation.

Usage sketch:
  >>> from milestone1_viz import *
  >>> df = load_your_clean_dataframe()
  >>> # Section 4
  >>> plot_shot_vs_goal_by_type(df, season="2019-20")
  >>> plot_goal_rate_vs_distance_by_season(df, seasons=["2018-19","2019-20","2020-21"])
  >>> plot_goal_pct_by_distance_and_type(df, season="2019-20")
  >>> # Section 5
  >>> export_interactive_shot_maps(df, seasons=["2016-17","2017-18","2018-19","2019-20","2020-21"],
  ...                              rink_image_path="./assets/rink_half.png")

"""
from __future__ import annotations
import os
import math
from typing import Iterable, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.colors import sequential


# Helpers:  cleaning

def _ensure_dirs():
    os.makedirs("./figures/simple", exist_ok=True)
    os.makedirs("./figures/advanced", exist_ok=True)


def normalise_attacking_right(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where x>=0 for all shots/goals (attacking to +x).
    If your ETL already flips by home/away & period, you may skip this.
    We simply mirror plays with x<0.
    """
    df = df.copy()
    mask = df["x"].astype(float) < 0
    df.loc[mask, "x"] = -df.loc[mask, "x"].astype(float)
    df.loc[mask, "y"] = -df.loc[mask, "y"].astype(float)
    return df


def distance_to_goal(x: np.ndarray, y: np.ndarray, goal_x: float = 89.0) -> np.ndarray:
    """Euclidean distance from (x,y) to the attacking goal at (goal_x, 0).
    Assumes coords are already normalised so the attacking net is to the right (+x).
    """
    dx = goal_x - x.astype(float)
    dy = y.astype(float)
    return np.sqrt(dx * dx + dy * dy)

# Simple visualisations (Matplotlib)

def plot_shot_vs_goal_by_type(df: pd.DataFrame, season: str, save=True) -> plt.Figure:
    """Figure 4.1 – Compare shot types across all teams and overlay goals

    Produces a grouped/overlaid bar plot of counts per shotType: shots vs goals.
    Returns the Matplotlib Figure.
    """
    _ensure_dirs()
    d = df[(df["season"] == season) & (df["eventType"].isin(["SHOT", "GOAL"]))].copy()
    d["shotType"] = d["shotType"].fillna("Unknown")

    shots = d[d["eventType"] == "SHOT"].groupby("shotType").size().rename("shots")
    goals = d[d["eventType"] == "GOAL"].groupby("shotType").size().rename("goals")
    g = pd.concat([shots, goals], axis=1).fillna(0).sort_values("shots", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    idx = np.arange(len(g))
    ax.bar(idx, g["shots"], width=0.8, alpha=0.6, label="Shots")
    ax.bar(idx, g["goals"], width=0.8, alpha=0.9, label="Goals")
    ax.set_xticks(idx)
    ax.set_xticklabels(g.index, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Shot vs Goal counts by shot type – {season}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(f"./figures/simple/shot_vs_goal_by_type_{season}.png", dpi=150)
    return fig


def plot_goal_rate_vs_distance_by_season(df: pd.DataFrame, seasons: Iterable[str],
                                         bin_width: float = 2.5, max_dist: float = 100.0,
                                         save=True) -> Dict[str, plt.Figure]:
    """Figure 4.1a – For each season, plot goal probability vs distance.

    Implementation details:
      • Normalise coords to attacking-right; compute distance to goal.
      • Bin distances; goal rate = goals / (shots + goals) per bin.
    Returns a dict of season -> Figure.
    """
    _ensure_dirs()
    figs = {}
    df_norm = normalise_attacking_right(df[df["eventType"].isin(["SHOT", "GOAL"])])
    df_norm["dist"] = distance_to_goal(df_norm["x"].values, df_norm["y"].values)

    bins = np.arange(0, max_dist + bin_width, bin_width)
    for season in seasons:
        d = df_norm[df_norm["season"] == season].copy()
        d["bin"] = pd.cut(d["dist"], bins=bins, right=False)
        by = d.groupby("bin")["eventType"].value_counts().unstack(fill_value=0)
        shots = by.get("SHOT", pd.Series(0, index=by.index))
        goals = by.get("GOAL", pd.Series(0, index=by.index))
        with np.errstate(divide='ignore', invalid='ignore'):
            goal_rate = goals / (shots + goals)
        centers = np.array([b.left + bin_width/2 for b in goal_rate.index.categories])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(centers, goal_rate.values, marker="o")
        ax.set_xlim(0, max_dist)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Distance to goal (ft)")
        ax.set_ylabel("P(goal | attempt)")
        ax.set_title(f"Goal probability vs distance – {season}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            fig.savefig(f"./figures/simple/goal_rate_vs_distance_{season}.png", dpi=150)
        figs[season] = fig
    return figs


def plot_goal_pct_by_distance_and_type(df: pd.DataFrame, season: str,
                                       bin_width: float = 2.5, max_dist: float = 100.0,
                                       min_count: int = 25, save=True) -> plt.Figure:
    """Figure 4.2 – % goals versus distance and shotType (heatmap)

    Compute P(goal|attempt) per (distance bin, shotType) and render as a heatmap.
    """
    _ensure_dirs()
    d = df[(df["season"] == season) & (df["eventType"].isin(["SHOT", "GOAL"]))].copy()
    d["shotType"] = d["shotType"].fillna("Unknown")
    d = normalise_attacking_right(d)
    d["dist"] = distance_to_goal(d["x"].values, d["y"].values)

    bins = np.arange(0, max_dist + bin_width, bin_width)
    d["bin"] = pd.cut(d["dist"], bins=bins, right=False)

    ct = d.groupby(["shotType", "bin"])['eventType'].value_counts().unstack(fill_value=0)
    shots = ct.get("SHOT", pd.Series(0, index=ct.index))
    goals = ct.get("GOAL", pd.Series(0, index=ct.index))
    tot = (shots + goals)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = (goals / tot).where(tot >= min_count)

    # Pivot to [shotType x dist_bin]
    pivot = pct.unstack("bin")
    # Prepare heatmap
    x_labels = [f"{b.left:.0f}-{b.right:.0f}" for b in pivot.columns]
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Distance bin (ft)")
    ax.set_ylabel("Shot type")
    ax.set_title(f"% Goals by distance and shot type – {season}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(goal | attempt)")
    plt.tight_layout()
    if save:
        fig.savefig(f"./figures/simple/goal_pct_by_distance_type_{season}.png", dpi=150)
    return fig

# Section 5 – Advanced visualisations (Plotly, HTML out)

RINK_X_MAX = 100.0  # logical half-length used in NHL feeds
RINK_Y_MAX = 42.5   # half-width


def _make_rink_background(fig: go.Figure, rink_image_path: Optional[str]) -> None:
    """If a rink image is provided, add it as a background. Otherwise, draw a minimalist rink."""
    if rink_image_path and os.path.exists(rink_image_path):
        fig.update_layout(images=[dict(
            source=rink_image_path,
            xref="x", yref="y",
            x=0, y=RINK_Y_MAX,
            sizex=RINK_X_MAX, sizey=2*RINK_Y_MAX,
            sizing="stretch", layer="below", opacity=0.8,
        )])
    else:
        fig.add_shape(type="rect", x0=0, y0=-RINK_Y_MAX, x1=RINK_X_MAX, y1=RINK_Y_MAX,
                      line=dict(width=2), layer="below")
        fig.add_shape(type="line", x0=RINK_X_MAX, y0=-RINK_Y_MAX, x1=RINK_X_MAX, y1=RINK_Y_MAX,
                      line=dict(width=3))
        fig.add_shape(type="circle", x0=89-6, y0=-6, x1=89+6, y1=6, line=dict(width=1), layer="below")


def _bivariate_kde_heatmap(x: np.ndarray, y: np.ndarray, bins: int = 75) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast 2D histogram as a KDE-like heatmap proxy in rink space."""
    H, xedges, yedges = np.histogram2d(x, y, bins=bins,
                                       range=[[0, RINK_X_MAX], [-RINK_Y_MAX, RINK_Y_MAX]])
    H = H.T  # transpose for plotly heatmap orientation
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    return H, xcenters, ycenters


def _shot_rate_per_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Compute league-average shot rate per hour heatmap grid.
    Simplifying assumptions: every game = 60 minutes; ignore PP/SH states (per spec).
    Returns a DataFrame with columns [x_center, y_center, rate_per_hour].
    """
    d = df.copy()
    # Count attempts per grid cell
    H, xcenters, ycenters = _bivariate_kde_heatmap(d["x"].values, d["y"].values, bins=75)
    attempts = H  # counts
    minutes = len(d["gameId"].unique()) * 60.0
    rate = attempts / (minutes / 60.0)  # attempts per hour
    grid = pd.DataFrame({
        "x": np.repeat(xcenters, len(ycenters)),
        "y": np.tile(ycenters, len(xcenters)),
        "rate_per_hour": rate.flatten()
    })
    return grid


def _team_differential_grid(df_norm: pd.DataFrame, team: str, league_grid: pd.DataFrame) -> pd.DataFrame:
    """Compute team shot rate per hour minus league average on same grid."""
    d = df_norm[df_norm["team"] == team]
    H, xcenters, ycenters = _bivariate_kde_heatmap(d["x"].values, d["y"].values, bins=75)
    attempts = H
    minutes = len(d["gameId"].unique()) * 60.0
    rate = attempts / (minutes / 60.0)

    team_grid = pd.DataFrame({
        "x": np.repeat(xcenters, len(ycenters)),
        "y": np.tile(ycenters, len(xcenters)),
        "rate_per_hour": rate.flatten()
    })
    # Align and compute difference
    merged = team_grid.merge(league_grid, on=["x", "y"], suffixes=("_team", "_league"))
    merged["diff"] = merged["rate_per_hour_team"] - merged["rate_per_hour_league"]
    return merged


def _heatmap_fig_from_grid(diff_grid: pd.DataFrame, title: str, rink_image_path: Optional[str]) -> go.Figure:
    cm = sequential.RdBu  # diverging scheme; Plotly handles color scaling
    vmax = np.nanmax(np.abs(diff_grid["diff"].values)) or 1.0
    fig = go.Figure()
    _make_rink_background(fig, rink_image_path)
    fig.add_trace(go.Heatmap(
        x=sorted(diff_grid["x"].unique()),
        y=sorted(diff_grid["y"].unique()),
        z=diff_grid.pivot(index="y", columns="x", values="diff").values,
        colorscale=cm,
        zmid=0.0,
        zmin=-vmax,
        zmax=vmax,
        colorbar=dict(title="Shots/60 vs league")
    ))
    fig.update_xaxes(range=[0, RINK_X_MAX], constrain="domain", title_text="x (ft)")
    fig.update_yaxes(range=[-RINK_Y_MAX, RINK_Y_MAX], scaleanchor="x", scaleratio=1, title_text="y (ft)")
    fig.update_layout(title=title, margin=dict(l=40, r=40, t=60, b=40))
    return fig


def export_interactive_shot_maps(df: pd.DataFrame, seasons: Iterable[str],
                                 rink_image_path: Optional[str] = None,
                                 filename_template: str = "./figures/advanced/shot_map_{season}.html",
                                 min_events_per_team: int = 200) -> Dict[str, str]:
    """For each season in seasons, export an interactive Plotly HTML figure with a dropdown to pick team.

    Spec simplifications per assignment:
      • Offensive zone only (attacking-right). We normalise and then clip to x in [0, 100].
      • Ignore PP/SH states.
      • Assume 60 min per game.

    Returns dict season -> output_html_path.
    """
    _ensure_dirs()
    outputs = {}

    base = df[df["eventType"].isin(["SHOT", "GOAL"])].copy()
    base = normalise_attacking_right(base)
    base = base[(base["x"] >= 0) & (base["x"] <= RINK_X_MAX) & (base["y"].between(-RINK_Y_MAX, RINK_Y_MAX))]

    for season in seasons:
        d = base[base["season"] == season]
        if d.empty:
            continue
        # League grid
        league_grid = _shot_rate_per_hour(d)

        # Build one trace per team as a frame; use dropdown to switch visibility
        teams = sorted([t for t, n in d["team"].value_counts().items() if n >= min_events_per_team])
        if not teams:
            teams = sorted(d["team"].unique())

        # Precompute grids and figures per team
        team_figs = {}
        for team in teams:
            diff_grid = _team_differential_grid(d, team, league_grid)
            team_figs[team] = diff_grid

        # Initialise with first team
        first_team = teams[0]
        fig = _heatmap_fig_from_grid(team_figs[first_team], title=f"Shot rate differential – {first_team} ({season})", rink_image_path=rink_image_path)

        # We'll replace z on selection via updatemenus
        buttons = []
        for team in teams:
            grid = team_figs[team]
            z = grid.pivot(index="y", columns="x", values="diff").values
            buttons.append(dict(
                label=team,
                method="update",
                args=[
                    {"z": [z]},
                    {"title": f"Shot rate differential – {team} ({season})"}
                ]
            ))

        fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                x=1.0, xanchor="right", y=1.1, yanchor="top",
                buttons=buttons,
                active=0,
                direction="down"
            )]
        )

        out_path = filename_template.format(season=season.replace("/", "-"))
        fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
        outputs[season] = out_path

    return outputs


# run as a script
if __name__ == "__main__":
    print("This module contains plotting utilities for Milestone 1 Sections 4 & 5.\n"
          "Import the functions and pass your cleaned events dataframe.")

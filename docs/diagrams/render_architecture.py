"""Render the FATEC architecture diagram (SVG + PNG) with matplotlib.

Usage:
    python docs/diagrams/render_architecture.py

Outputs (in the same directory):
    architecture.svg  - committed, referenced by the main README
    architecture.png  - local preview (git-ignored by the global *.png rule)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Ellipse, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

# Canvas (data coordinates) -----------------------------------------------
X_MAX, Y_MAX = 150, 100
FIG_W, FIG_H = 15, 8.6

# Palette ------------------------------------------------------------------
C_CLIENT = ("#E8EEF7", "#3B5B92")
C_API = ("#3B5B92", "#2C4570")
C_TASK = ("#DCE8F5", "#3B5B92")
C_PROC = ("#E2EFDA", "#548235")
C_FETCH = ("#FDF2E3", "#D68910")
C_RESULT = ("#FFF6DD", "#B7950B")
C_OUT = ("#EDEDED", "#7F8C8D")
C_DB = ("#E8DAEF", "#7D3C98")
C_ARROW = "#555555"

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=150)
ax.set_xlim(0, X_MAX)
ax.set_ylim(0, Y_MAX)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_title("FATEC — asynchronous task pipeline", loc="left", fontsize=14, pad=14)


def box(cx, cy, w, h, text, fc, ec, lw=1.3, ls="-", radius=1.0,
        fontsize=8.5, text_color="#222222", weight="normal"):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       fc=fc, ec=ec, lw=lw, ls=ls,
                       mutation_aspect=1.16, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, zorder=4, linespacing=1.4, weight=weight)
    return p


def arrow(p1, p2, color=C_ARROW, lw=1.5, ls="-", style="-|>"):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0.8,
                                mutation_scale=13),
                zorder=2)


# Top: client + API ---------------------------------------------------------
box(62, 94.5, 24, 7, "Client", *C_CLIENT, fontsize=10, weight="bold")
box(62, 84.5, 46, 8, "FastAPI — app.py\nREST endpoints + background task dispatch",
    *C_API, text_color="white", fontsize=9, weight="bold")
arrow((62, 91.0), (62, 89.0))
ax.text(64.5, 89.9, "POST /execute · /clustering · /ewh_optimization",
        fontsize=7.5, color="#444444", va="center", zorder=4)

# Step columns --------------------------------------------------------------
box(22, 72.5, 36, 7, "Step 1 · background task\nrun_consumers()  (main_elasticity.py)", *C_TASK, fontsize=8)
box(62, 72.5, 36, 7, "Step 2 · background task\nrun_clustering()  (main_clusters.py)", *C_TASK, fontsize=8)
box(102, 72.5, 38, 7, "Step 3 · background task\nrun_ewh_optimization()  (main_ewh.py)", *C_TASK, fontsize=8)
arrow((42, 80.5), (31, 76.5))
arrow((62, 80.5), (62, 76.5))
arrow((81, 80.5), (96, 76.5))

# Processing rows -----------------------------------------------------------
box(22, 60.5, 34, 10, "DataPreprocessor (processing/)\nraw data + Open-Meteo weather", *C_PROC)
box(62, 60.5, 36, 10, "HTTP fetch of parent task results\nGET /result/{task_id}\nvia API_BASE_URL",
    *C_FETCH, ls=(0, (5, 3)))
box(102, 60.5, 38, 10, "HTTP fetch of clustering results\nGET /clustering_result/{id}\nvia API_BASE_URL",
    *C_FETCH, ls=(0, (5, 3)))

box(22, 45.5, 34, 11, "ElasticityModel (causality/)\ncausal DML — econml + XGBoost\nper consumer or aggregated", *C_PROC)
box(62, 45.5, 36, 11, "KMeans + silhouette score\n+ synthetic EWH profiles\n(clusters by consumption\n& elasticity profiles)", *C_PROC)
box(102, 45.5, 38, 11, "EWH MILP optimization\n(PuLP / CBC — ewh/ewh_flex)\nper cluster mean profile", *C_PROC)

arrow((22, 69.0), (22, 65.5))
arrow((62, 69.0), (62, 65.5))
arrow((102, 69.0), (102, 65.5))
arrow((22, 55.5), (22, 51.5))
arrow((62, 55.5), (62, 51.5))
arrow((102, 55.5), (102, 51.5))
arrow((22, 40.0), (22, 36.0))
arrow((62, 40.0), (62, 36.0))
arrow((102, 40.0), (102, 36.0))

# Result row ----------------------------------------------------------------
box(22, 31.5, 30, 9, "Elasticity profiles,\nranking, p-values", *C_RESULT)
box(62, 31.5, 30, 9, "Consumer clusters\n(members + mean profiles)", *C_RESULT)
box(102, 31.5, 30, 9, "Optimized EWH calendars\nenergy & cost savings", *C_RESULT)

# Result collector bus --------------------------------------------------------
BUS_Y = 25.5
for x in (22, 62, 102):
    ax.add_line(Line2D([x, x], [27.0, BUS_Y], color=C_ARROW, lw=1.2, ls=":", zorder=2))
ax.add_line(Line2D([22, 102], [BUS_Y, BUS_Y], color=C_ARROW, lw=1.5, zorder=2))

# Storage row -----------------------------------------------------------------
box(50, 15, 48, 8, "results/<timestamp>_<task_id>/ — logs · plots · JSON",
    *C_OUT, fontsize=8)
arrow((44, BUS_Y), (44, 19.5))

# PostgreSQL cylinder
db_x0, db_x1, db_y0, db_y1, db_e = 114, 140, 11, 33, 3.0
db_cx = (db_x0 + db_x1) / 2
db_w = db_x1 - db_x0
ax.add_patch(Rectangle((db_x0, db_y0), db_w, db_y1 - db_y0,
                       fc=C_DB[0], ec="none", zorder=3))
ax.add_patch(Ellipse((db_cx, db_y0), db_w, 2 * db_e, fc=C_DB[0], ec="none", zorder=3))
ax.add_patch(Ellipse((db_cx, db_y1), db_w, 2 * db_e, fc=C_DB[0], ec=C_DB[1], lw=1.3, zorder=4))
ax.add_patch(Arc((db_cx, db_y0), db_w, 2 * db_e, theta1=180, theta2=360,
                 ec=C_DB[1], lw=1.3, zorder=4))
ax.text(db_cx, (db_y0 + db_y1) / 2 - 2.2, "PostgreSQL\ntasks\nclustering_tasks\newh_tasks",
        ha="center", va="center", fontsize=8.5, color="#222222", zorder=4)
ax.text(db_cx, db_y0 - db_e - 2.2, "database", ha="center", fontsize=7.5,
        style="italic", color=C_DB[1])
arrow((102, BUS_Y), (db_x0, BUS_Y))
ax.text(106, BUS_Y - 1.9, "status + JSONB results", fontsize=7.5, ha="center", color="#444444")

# API -> DB persistence (dashed elbow) ----------------------------------------
db_top = db_y1 + db_e
ax.add_line(Line2D([85, db_cx], [84.5, 84.5], color=C_DB[1], lw=1.3,
                   ls=(0, (5, 3)), zorder=2))
ax.add_line(Line2D([db_cx, db_cx], [84.5, db_top + 1.8], color=C_DB[1], lw=1.3,
                   ls=(0, (5, 3)), zorder=2))
arrow((db_cx, db_top + 1.8), (db_cx, db_top + 0.4), color=C_DB[1], ls=(0, (5, 3)))
ax.text(db_cx + 1.2, 62, "persist task · store result", fontsize=7.5, color=C_DB[1],
        rotation=90, va="center", ha="left")

# Legend -----------------------------------------------------------------------
legend_items = [
    mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.15",
                            fc=C_API[0], ec=C_API[1], lw=1.2), "service / API",
    mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.15",
                            fc=C_TASK[0], ec=C_TASK[1], lw=1.2), "async background task",
    mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.15",
                            fc=C_PROC[0], ec=C_PROC[1], lw=1.2), "processing step",
    mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.15",
                            fc=C_FETCH[0], ec=C_FETCH[1], lw=1.2, ls="--"), "HTTP fetch (API_BASE_URL)",
    mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.15",
                            fc=C_RESULT[0], ec=C_RESULT[1], lw=1.2), "analysis results",
    mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.15",
                            fc=C_DB[0], ec=C_DB[1], lw=1.2), "database / artifacts",
]
handles, labels = zip(*[ (legend_items[i], legend_items[i + 1]) for i in range(0, len(legend_items), 2) ])
ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=8,
          bbox_to_anchor=(0.005, 1.0), ncol=1, handletextpad=0.6)

svg_path = os.path.join(HERE, "architecture.svg")
png_path = os.path.join(HERE, "architecture.png")
fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
fig.savefig(png_path, bbox_inches="tight", dpi=200, facecolor="white")
print(f"Saved: {svg_path}")
print(f"Saved: {png_path}")

import numpy as np
import matplotlib.pyplot as plt

# ---- raw table (include BE1D, we'll drop it) ----
#            name           E_PT      E_NB      E_SC       GCR
raw = [
    ("CFD1D",        0.1247,  0.145928, 0.06102,  0.237912877),
    ("BE1D",         0.48774, 0.56384,  0.03961,  0.14516529),
    ("CFD2D",        0.09604, 0.10238,  0.05318,  0.1288617886),
    ("CFD3D",        0.1464,  0.15688,  0.09819,  0.1785653433),
    ("MHD3D (V)",    0.82421, 0.88503,  0.3149,   0.1066774244),
    ("CFD3D-TURB",   0.39639, 0.42481,  0.08297,  0.083138313032),
    ("TGC3D (V)",    0.45407, 0.48968,  0.16366,  0.1092264278),
]

# ---- drop BE1D and split columns ----
rows = [r for r in raw if r[0] != "BE1D"]
labels = [r[0] for r in rows]
E_PT   = [r[1] for r in rows]
E_NB   = [r[2] for r in rows]
E_SC   = [r[3] for r in rows]
GCR    = [r[4] for r in rows]

# ---- grouped bar chart with specific colors + savefig ----
x = np.arange(len(labels))
w = 0.2
colors = ['C0', 'C1', 'C2', 'C3']  # E_PT, E_NB, E_SC, GCR

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.bar(x - 1.5*w, E_PT, width=w, label="E_PT", color=colors[0])
ax.bar(x - 0.5*w, E_NB, width=w, label="E_NB", color=colors[1])
ax.bar(x + 0.5*w, E_SC, width=w, label="E_SC", color=colors[2])
ax.bar(x + 1.5*w, GCR,  width=w, label="GCR", color=colors[3])

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Value")
ax.set_title("First 4 columns per dataset (excluding BE1D)")
ax.legend()
fig.tight_layout()

# save as PNG at 300 dpi
fig.savefig("transfer_study_bar.png", dpi=300, bbox_inches="tight")

plt.show()
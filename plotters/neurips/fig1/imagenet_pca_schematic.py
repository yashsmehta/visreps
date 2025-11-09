# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----- Nature-like styling -----
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8.5, "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
    "font.family": "sans-serif", "axes.linewidth": 0.8,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.frameon": False,
})

# ----- Config -----
SEED = 7
N_POINTS, D, N_CLASSES = 10_000, 50, 50
PTS_PER_C = N_POINTS // N_CLASSES
ELLIPSE_A, ELLIPSE_B = 4.0, 6.0
SIGMA_CLUSTER = 0.50
TILT_ANGLE = np.pi/6
MARKER_SIZE, ALPHA = 4, 0.8
rng = np.random.default_rng(SEED)

# Okabe–Ito colors (high contrast)
CBLUE   = "#0072B2"
CVERMIL = "#D55E00"
CGREEN  = "#009E73"
CMAG    = "#CC79A7"

def random_orthonormal(d, k=2, rng=None):
    if rng is None: rng = np.random.default_rng()
    Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return Q[:, :k]

def palette(n):
    base = [CBLUE, CVERMIL, CGREEN, CMAG, "#56B4E9", "#E69F00"]
    return [base[i % len(base)] for i in range(n)]

def strip_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

# ----- Data -----
U2 = random_orthonormal(D, k=2, rng=rng)

centroids_2d = []
for _ in range(N_CLASSES):
    r = np.sqrt(rng.uniform(0, 1))
    th = rng.uniform(0, 2*np.pi)
    centroids_2d.append([ELLIPSE_A*r*np.cos(th), ELLIPSE_B*r*np.sin(th)])
centroids_2d = np.array(centroids_2d)
R_tilt = np.array([[np.cos(TILT_ANGLE), -np.sin(TILT_ANGLE)],
                   [np.sin(TILT_ANGLE),  np.cos(TILT_ANGLE)]])
centroids_2d = centroids_2d @ R_tilt.T

X_list, y_list = [], []
for k in range(N_CLASSES):
    Z = rng.normal(scale=SIGMA_CLUSTER, size=(PTS_PER_C, 2))
    pts2 = centroids_2d[k] + Z
    pts50 = pts2 @ U2.T + rng.normal(scale=0.02, size=(PTS_PER_C, D))
    X_list.append(pts50)
    y_list.append(np.full(PTS_PER_C, k, np.int32))
X = np.vstack(X_list); y = np.concatenate(y_list)
X = X @ random_orthonormal(D, k=D, rng=rng)

# ----- PCA shared coords -----
X2 = PCA(n_components=2, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
pc1, pc2 = X2[:, 0], X2[:, 1]
med1, med2 = np.median(pc1), np.median(pc2)

# ----- Figure -----
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=300, constrained_layout=True)

# (a) 50 proxy classes
ax = axes[0]
cols = palette(N_CLASSES)
for k in range(N_CLASSES):
    m = (y == k)
    ax.scatter(X2[m,0], X2[m,1], s=MARKER_SIZE, alpha=ALPHA, c=[cols[k]],
               edgecolors='none', rasterized=True)
strip_axes(ax)
ax.set_title("ImageNet 1K Classes", pad=2.5)

# (b) 2-class PC1 median split; PC1 arrow through (x, y = med2)
ax = axes[1]
mask = (pc1 >= med1)
ax.scatter(X2[~mask,0], X2[~mask,1], s=MARKER_SIZE, alpha=ALPHA, c=CBLUE,
           edgecolors='none', rasterized=True)
ax.scatter(X2[mask,0],  X2[mask,1],  s=MARKER_SIZE, alpha=ALPHA, c=CVERMIL,
           edgecolors='none', rasterized=True)
strip_axes(ax)
# compute range after plotting
xmin, xmax = X2[:,0].min(), X2[:,0].max()
Lx = 0.42 * (xmax - xmin)
cx = (xmin + xmax) / 2.0
ax.annotate("", xy=(cx+Lx, med2), xytext=(cx-Lx, med2),
            arrowprops=dict(arrowstyle="<->", lw=1.8, color="black"))
ax.text(cx+Lx*1.05, med2, "PC1", fontsize=9, weight="bold", va="center", ha="left")
ax.set_title("ImageNet 2 Classes", pad=2.5)

# (c) 4-class quadrant split; PC1/PC2 arrows cross at (med1, med2)
ax = axes[2]
q00 = (pc1 <  med1) & (pc2 <  med2)
q10 = (pc1 >= med1) & (pc2 <  med2)
q01 = (pc1 <  med1) & (pc2 >= med2)
q11 = (pc1 >= med1) & (pc2 >= med2)
quad_colors = [CBLUE, CVERMIL, CGREEN, CMAG]  # BL, BR, TL, TR
for msk, cc in zip([q00, q10, q01, q11], quad_colors):
    ax.scatter(X2[msk,0], X2[msk,1], s=MARKER_SIZE, alpha=ALPHA, c=cc,
               edgecolors='none', rasterized=True)
strip_axes(ax)
xmin, xmax = X2[:,0].min(), X2[:,0].max()
ymin, ymax = X2[:,1].min(), X2[:,1].max()
Lx = 0.42 * (xmax - xmin); Ly = 0.42 * (ymax - ymin)
ax.annotate("", xy=(med1+Lx, med2), xytext=(med1-Lx, med2),
            arrowprops=dict(arrowstyle="<->", lw=1.6, color="black"))
ax.annotate("", xy=(med1, med2+Ly), xytext=(med1, med2-Ly),
            arrowprops=dict(arrowstyle="<->", lw=1.6, color="black"))
ax.text(med1+Lx*1.05, med2, "PC1", fontsize=8.5, weight="bold", va="center", ha="left")
ax.text(med1, med2+Ly*1.05, "PC2", fontsize=8.5, weight="bold", va="bottom", ha="center")
ax.set_title("ImageNet 4 Classes", pad=2.5)

# Panel letters
for i, ax in enumerate(axes):
    ax.text(0.01, 0.98, chr(ord('a') + i), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, fontweight="bold")

# Save
out_png = "plotters/neurips/fig1/schematic_imagenet_pca.png"
fig.savefig(out_png, bbox_inches="tight")
print(f"Saved:\n  {out_png}")
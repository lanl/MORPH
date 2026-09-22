import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA   
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def sensitivity_analyser(images, scalars, params, save_dir=None):

    N,H,W,C = images.shape
    X = images.reshape(N, -1).astype(np.float32)
    Y = params.astype(np.float32)
    S = scalars.astype(np.float32)
    print(f"X.shape: {X.shape}, Y.shape: {Y.shape}, S.shape: {S.shape}")

    # standardize X and Y for comparable coefficient scales
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-8)

    # --- PCA compress images to PCA components ---
    pca_components = 32
    pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=0)
    Z = pca.fit_transform(X)                    # (N, 10) PCA scores per image
    Zz = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)    # standardize PC scores

    print("Z.shape (PC scores):", Z.shape)
    print("Explained variance (first 8):", pca.explained_variance_ratio_.sum())
    
    # --- Ridge regression of PCA scores against params ---
    # sensitivity analysis as the ratio of output (images) to inputs (params)
    reg = Ridge(alpha=1.0, solver="lsqr")
    reg.fit(Yz, Zz)

    coef = reg.coef_   # (pca_components, 5): each row = PCA, each col = param
    print("coef.shape (PC x param):", coef.shape)

    # plot influence heatmap (signed effects)
    param_names = [f"param {i}" for i in range(Y.shape[1])]
    pc_names = [f"PC{i+1}" for i in range(pca_components)]

    v = np.max(np.abs(coef))
    plt.figure(figsize=(6, 8))
    im = plt.imshow(coef, aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)
    plt.xticks(range(len(param_names)), param_names, rotation=45, ha="right")
    plt.yticks(range(len(pc_names)), pc_names)
    plt.title("Param → PCA component influence (Ridge coefficients)")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Δ(PC z-score) per Δ(param z-score)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sensitivity_analysis.png'), dpi = 300)
    plt.close()

    # --- combine scalars and image PCs for sensitivity analysis ---
    # combine features: [PCs | scalars]
    F = np.concatenate([Zz, Sz], axis=1)           # (N, 16+15)

    # standardize targets (params)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)

    # fit: (images PCs + scalars) -> params
    reg = Ridge(alpha=1.0, solver="lsqr")
    reg.fit(F, Yz)

    coef = reg.coef_   # (5, 31): rows=params, cols=[PC1..PC16 | scalar0..scalar14]
    print("coef.shape (param x features):", coef.shape)

    # --- plot coefficient heatmap ---
    param_names = [f"param {i}" for i in range(Y.shape[1])]
    feat_names = [f"PC{i+1}" for i in range(pca_components)] + [f"scalar {i}" for i in range(S.shape[1])]

    v = np.max(np.abs(coef))
    plt.figure(figsize=(14, 4))
    im = plt.imshow(coef, aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)
    plt.xticks(range(len(feat_names)), feat_names, rotation=45, ha="right")
    plt.yticks(range(len(param_names)), param_names)
    plt.axvline(pca_components - 0.5, color="k", linewidth=1)  # split PCs vs scalars
    plt.title("Combined influence: (image PCs + scalars) → params (Ridge coefficients)")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Δ(param z-score) per Δ(feature z-score)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sensitivity_analysis_combined_heatmap.png'), dpi = 300)
    plt.close()

    # --- summarize how much each group contributes per param ---
    pc_infl = np.linalg.norm(coef[:, :pca_components], axis=1)
    sc_infl = np.linalg.norm(coef[:, pca_components:], axis=1)

    plt.figure(figsize=(6, 3))
    x = np.arange(len(param_names))
    plt.bar(x - 0.2, pc_infl, width=0.4, label="image PCs")
    plt.bar(x + 0.2, sc_infl, width=0.4, label="scalars")
    plt.xticks(x, param_names)
    plt.title("Influence magnitude by feature group (L2 of coefficients)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sensitivity_analysis_combined.png'), dpi = 300)
    plt.close()


    # --- R2 comparison ---
    def r2_per_param(F, Yz, alpha=1.0):
        Ftr, Fte, Ytr, Yte = train_test_split(F, Yz, test_size=0.2, random_state=0)
        reg = Ridge(alpha=alpha, solver="lsqr")
        reg.fit(Ftr, Ytr)
        Ypred = reg.predict(Fte)
        return np.array([r2_score(Yte[:,j], Ypred[:,j]) for j in range(Yz.shape[1])])

    F_both = np.concatenate([Zz, Sz], axis=1)

    r2_pc   = r2_per_param(Zz, Yz)
    r2_sc   = r2_per_param(Sz, Yz)
    r2_both = r2_per_param(F_both, Yz)

    print("R2 PCs only:     ", r2_pc)
    print("R2 scalars only: ", r2_sc)
    print("R2 both:         ", r2_both)
    print("ΔR2 (add PCs on top of scalars):", r2_both - r2_sc)
    print("ΔR2 (add scalars on top of PCs):", r2_both - r2_pc)
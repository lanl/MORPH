def custom_patchify_3d(x, patch_size):
    """
    Flexible 3D patch extraction with debug prints:
      - patch_size: int or (pD, pH, pW)
      - if any axis < requested patch_size, we take exactly 1 patch along that axis of full size
    """
    # ——— 0) input shape ———
    B, C, D, H, W = x.shape
    # print(f"[patchify] input shape:   B={B}, C={C}, D={D}, H={H}, W={W}")

    # ——— 1) unpack patch size ———
    if isinstance(patch_size, (tuple, list)):
        pz, py, px = patch_size
    else:
        pz = py = px = patch_size
    # print(f"[patchify] requested patch size: pz={pz}, py={py}, px={px}")

    # ——— 2) adjust for small axes ———
    pz = pz if D >= pz else D
    py = py if H >= py else H
    px = px if W >= px else W
    # print(f"[patchify] adjusted patch size: pz={pz}, py={py}, px={px}")

    # ——— 3) compute number of patches per axis ———
    nz, ny, nx = D // pz, H // py, W // px
    assert D % pz == 0 and H % py == 0 and W % px == 0, \
           f"Dimensions {(D,H,W)} must be divisible by patches {(pz,py,px)}"
    # print(f"[patchify] number of patches: nz={nz}, ny={ny}, nx={nx} (total={nz*ny*nx})")

    # ——— 4) reshape into patches ———
    x = x.reshape(B, C, nz, pz, ny, py, nx, px)
    # print(f"[patchify] after reshape:     {x.shape}  (B,C,nz,pz,ny,py,nx,px)")

    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    # print(f"[patchify] after permute:     {x.shape}  (B,nz,ny,nx,C,pz,py,px)")

    x = x.reshape(B, nz * ny * nx, C * pz * py * px)
    # print(f"[patchify] final output shape: {x.shape}  (B, num_patches, patch_vol * C)")

    return x

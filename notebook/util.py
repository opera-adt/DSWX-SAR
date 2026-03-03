
from pathlib import Path
import requests
from tqdm import tqdm
import os, re, warnings
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import rasterio as rio
import h5py
import matplotlib.pyplot as plt


def fetch_to_cache(url: str, cache_dir: str | Path = "./_cache",
                   filename: str | None = None, headers: dict | None = None,
                   force: bool = False, chunk_size: int = 2**22) -> Path:
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    name = filename or url.split("?", 1)[0].split("/")[-1]
    out = cache_dir / name
    if out.exists() and not force:
        return out

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(out, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {name}") as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return out


def robust_percentiles(a, lo=2, hi=98):
    a = a[np.isfinite(a)]
    if a.size == 0: return (0.0, 1.0)
    return np.percentile(a, lo), np.percentile(a, hi)

def is_discrete(a, max_unique=32):
    return np.issubdtype(a.dtype, np.integer) and np.unique(a[:: max(1, a.size//100000)]).size <= max_unique

def plot_raster(arr, title, cmap=None):
    plt.figure(figsize=(8,6))
    if not is_discrete(arr):
        vmin, vmax = robust_percentiles(arr)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            im = plt.imshow(arr, vmin=vmin, vmax=vmax)
        else:
            im = plt.imshow(arr)
    else:
        if cmap is None:
            im = plt.imshow(arr, interpolation="nearest")
        else:
            im = plt.imshow(arr, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar(im, shrink=0.7)
    plt.tight_layout()
    plt.show()

def print_stats(arr, name):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"{name}: all NaN"); return
    mn, p2, med, p98, mx = np.min(finite), *np.percentile(finite,[2,50,98]), np.max(finite)
    print(f"{name}: min={mn:.4f}  p2={p2:.4f}  med={med:.4f}  p98={p98:.4f}  max={mx:.4f}  n={finite.size}")


priority = [r".*/C11.*", r".*/C22.*", r".*/C12.*real.*", r".*/C12.*imag.*",
            r".*/gamma0.*", r".*/sigma0.*", r".*/coh.*", r".*/corr.*"]

def list_2d_datasets(h):
    out = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
            out.append(name)
    h.visititems(lambda n,o: visitor(n,o))
    return out

import matplotlib.pyplot as plt
import rasterio as rio
from matplotlib.colors import ListedColormap

def _robust_p2p98(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None
    p2, p98 = np.percentile(x, (2, 98))
    return (p2, p98) if np.isfinite(p2) and np.isfinite(p98) and p2 < p98 else (None, None)


def plot_wtr(tif):
    with rio.open(tif) as src:
        title = tif.name

        # Try to read an embedded colortable on band 1 (paletted single-band TIFFs)
        try:
            cmap_dict = src.colormap(1)
        except Exception:
            cmap_dict = None

        if cmap_dict:  # --- paletted single-band ---
            arr = src.read(1)

            # Build LUT (RGBA 0..1) and alpha lookup
            max_key = max(cmap_dict.keys())
            lut = np.zeros((max_key + 1, 4), dtype=float)
            alpha_lut = np.zeros(max_key + 1, dtype=np.uint8)
            for k, (r, g, b, a) in cmap_dict.items():
                lut[k] = np.array([r, g, b, a], dtype=float) / 255.0
                alpha_lut[k] = a
            cmap = ListedColormap(lut)

            # Mask: nodata, out-of-range indices, or palette alpha==0
            mask = np.zeros(arr.shape, dtype=bool)
            if src.nodata is not None:
                mask |= (arr == src.nodata)
            in_range = (arr >= 0) & (arr <= max_key)
            mask |= ~in_range
            alpha_zero = np.zeros_like(mask, dtype=bool)
            valid = in_range
            alpha_zero[valid] = (alpha_lut[arr[valid]] == 0)
            mask |= alpha_zero

            arr_ma = np.ma.masked_where(mask, arr)

            plt.figure(figsize=(8, 6))
            im = plt.imshow(
                arr_ma,
                cmap=cmap,
                vmin=0, vmax=cmap.N - 1,           # direct index→palette mapping
                interpolation="nearest",
            )
            # Colorbar ticks at present classes (thin if too many)
            classes = np.unique(arr_ma.compressed()).astype(int)
            if classes.size > 0:
                cbar = plt.colorbar(im, fraction=0.046, pad=0.02)
                if classes.size <= 20:
                    cbar.set_ticks(classes)
                    cbar.set_ticklabels([str(v) for v in classes])
            plt.title(f"{title} (paletted)")
            plt.axis("off")
            plt.tight_layout(); plt.show()

        elif src.count >= 3:  # --- true RGB (3+ bands) ---
            # Read first 3 bands and stretch per band; DO NOT pass cmap
            rgb = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1).astype("float32")
            for i in range(3):
                vmin, vmax = _robust_p2p98(rgb[..., i])
                if vmin is not None and vmax is not None and vmax > vmin:
                    rgb[..., i] = (rgb[..., i] - vmin) / max(vmax - vmin, 1e-12)
            rgb = np.clip(rgb, 0, 1)

            plt.figure(figsize=(8, 6))
            plt.imshow(rgb)  # no cmap for RGB
            plt.title(f"{title} (RGB 1/2/3)")
            plt.axis("off")
            plt.tight_layout(); plt.show()

        else:  # --- single-band without palette (grayscale fallback) ---
            arr = src.read(1).astype("float32")
            vmin, vmax = _robust_p2p98(arr)
            plt.figure(figsize=(8, 6))
            plt.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)
            plt.title(f"{title} (single-band grayscale)")
            plt.colorbar(fraction=0.046, pad=0.02)
            plt.axis("off")
            plt.tight_layout(); plt.show()

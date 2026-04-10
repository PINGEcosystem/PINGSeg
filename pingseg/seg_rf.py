'''
Copyright (c) 2025 Cameron S. Bodine

Adapted from Segmentation Gym: https://github.com/Doodleverse/segmentation_gym

'''

#########
# Imports
import os
import tempfile
import pandas as pd

import base64
import io
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from PIL import Image
import cv2
import rasterio as rio
from rasterio.transform import from_bounds as rio_from_bounds
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

CLASS_MAP_DEFAULT = {"0": "background", "1": "SAV"}

def decode_mask_png_base64(b64_png: str) -> np.ndarray:
    """Decode base64 PNG to a 2D numpy array of uint8 pixel values."""
    data = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(data))
    arr = np.array(img)
    # If PNG is RGB(A) but encoded classes in single channel, try to reduce:
    if arr.ndim == 3:
        # If RGBA, pick first channel (assumes paletted or identical channels)
        if arr.shape[2] == 4 or arr.shape[2] == 3:
            # If paletted image (P mode), pillow would produce 2D, but handle RGB anyway:
            arr = arr[..., 0]
    return arr

def contours_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """Find external contours from a binary mask (0/1). Returns list of Nx2 int arrays."""
    # OpenCV expects uint8 0/255
    im = (mask.astype(np.uint8) * 255)
    # findContours modifies input, copy it
    im_cpy = im.copy()
    contours, hierarchy = cv2.findContours(im_cpy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # convert contours to Nx2 arrays
    out = [cnt.reshape(-1, 2) for cnt in contours if cnt.reshape(-1,2).shape[0] >= 3]
    return out

def polygon_from_contour(contour: np.ndarray, min_area: float=1.0, simplify_tolerance: float=0.5) -> Optional[Polygon]:
    """Convert contour (Nx2) to shapely Polygon, filter by area and simplify."""
    poly = Polygon(contour)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.is_empty:
        return None
    if poly.area < min_area:
        return None
    if simplify_tolerance and simplify_tolerance > 0:
        poly = poly.simplify(simplify_tolerance)
        if poly.is_empty:
            return None
    return poly

def mask_json_to_geojson(mask_json: Dict[str, Any],
                         min_area: float = 5.0,
                         simplify_tolerance: float = 0.5,
                         include_background: bool = False) -> Dict[str, Any]:
    """
    Convert the segmentation JSON (with a base64 PNG in 'segmentation_mask' and 'class_map')
    to a GeoJSON-like FeatureCollection (in image pixel coordinates).
    """
    # b64 = mask_json.segmentation_mask
    b64 = mask_json.get("segmentation_mask")
    if b64 is None:
        raise ValueError("No 'segmentation_mask' key found in input JSON.")
    class_map = mask_json.get("class_map", CLASS_MAP_DEFAULT)
    # decode mask (2D array)
    arr = decode_mask_png_base64(b64)
    # If image in JSON has width/height, ensure shape matches or warn
    img_info = mask_json.get("image", {})
    h = img_info.get("height")
    w = img_info.get("width")
    if h and w:
        if arr.shape[0] != h or arr.shape[1] != w:
            # If arr has extra channels, decode_mask already tried to remove them, else warn
            print(f"Warning: decoded mask shape {arr.shape} doesn't match provided image WxH {(w,h)}")
    features = []
    unique_ids = np.unique(arr)
    for uid in unique_ids:
        if uid == 0 and not include_background:
            continue
        # Create binary mask for this id
        mask = (arr == uid).astype(np.uint8)
        if mask.sum() == 0:
            continue
        # find contours
        contours = contours_from_mask(mask)
        polygons = []
        for cnt in contours:
            poly = polygon_from_contour(cnt, min_area=min_area, simplify_tolerance=simplify_tolerance)
            if poly is not None:
                polygons.append(poly)
        if not polygons:
            continue
        # Merge polygons for the class to a MultiPolygon-like structure
        if len(polygons) == 1:
            geom = mapping(polygons[0])
        else:
            union = unary_union(polygons)
            geom = mapping(union)
        properties = {
            "class_id": int(uid),
            "class_name": class_map.get(str(int(uid)), str(int(uid))),
            # optionally include counts/area
            "pixel_area": int((arr == uid).sum())
        }
        feat = {
            "type": "Feature",
            "geometry": geom,
            "properties": properties
        }
        features.append(feat)
    fc = {"type": "FeatureCollection", "features": features}
    return fc


def _is_base64_png(s: str) -> bool:
    """Return True if s is a base64 string that decodes to a PNG signature.

    Accepts optional data URI prefixes like 'data:image/png;base64,...'.
    """
    if not isinstance(s, str):
        return False
    # strip data URI prefix
    if s.startswith("data:") and ";base64," in s:
        s = s.split(";base64,", 1)[1]
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception:
        # Some implementations don't use validate=True; try a permissive decode
        try:
            raw = base64.b64decode(s)
        except Exception:
            return False
    # PNG signature
    return isinstance(raw, (bytes, bytearray)) and raw[:8] == b"\x89PNG\r\n\x1a\n"


def _find_segmentation_mask(obj: Any) -> Optional[Tuple[str, List[str]]]:
    """Recursively search obj (dict/list/scalar) for a base64 PNG string.

    Returns a tuple (b64_string, path_list) where path_list describes the keys/indexes
    to reach the value, or None if not found.
    """
    from collections import deque

    # BFS stack of (current_obj, path)
    q = deque([(obj, [])])
    visited = set()
    while q:
        cur, path = q.popleft()
        # avoid revisiting mutable containers
        try:
            vid = id(cur)
            if vid in visited:
                continue
            visited.add(vid)
        except Exception:
            pass

        # If it's a string, test it
        if isinstance(cur, str):
            if _is_base64_png(cur):
                return cur, path
            # also accept data URLs which may include spaces/newlines
            s = cur.strip()
            if _is_base64_png(s):
                return s, path
            continue

        # If it's a dict, enqueue values
        if isinstance(cur, dict):
            for k, v in cur.items():
                # common key names to prioritize
                if isinstance(k, str) and k.lower() in ("segmentation_mask", "mask", "segmentation", "segmentation_png", "encoded_mask", "mask_base64", "segmentation_base64"):
                    if isinstance(v, str) and _is_base64_png(v):
                        return v, path + [k]
                q.append((v, path + [k]))
            continue

        # If it's a list/tuple, enqueue elements
        if isinstance(cur, (list, tuple)):
            for i, v in enumerate(cur):
                q.append((v, path + [str(i)]))
            continue

        # other scalars ignored
    return None





def _prepare_tile_for_inference(tile_path: str):
    """
    Read a tile (1-band or 3-band GeoTIFF) and produce a clean 3-band PNG suitable
    for Roboflow inference.

    For 1-band tiles (grayscale sonar), the single channel is replicated to RGB.
    For 3-band tiles, the first three bands are used directly.

    Returns
    -------
    (infer_path, is_temp) : str, bool
        infer_path  – path to the PNG to send to the model.
        is_temp     – True when infer_path is a temporary file that the caller must delete.
    """
    with rio.open(tile_path) as src:
        band_count = src.count
        data = src.read()  # (bands, H, W)

    if band_count == 1:
        gray = data[0].astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)  # (H, W, 3)
    else:
        rgb = np.moveaxis(data[:3], 0, -1).astype(np.uint8)  # (H, W, 3)

    img = Image.fromarray(rgb, mode='RGB')
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    img.save(tmp.name)
    return tmp.name, True


#=======================================================================
def seg_rf_folder(imgDF: pd.DataFrame,
                  my_api_key: str,
                  my_model_name: str,
                  my_model_ver: str,
                  out_dir: str,
                  minPatchSize: float = 0,
                  batch_size: int = 8,
                  threadCnt: int = 1):
    """
    Run Roboflow semantic-segmentation inference on each tile in *imgDF* and save
    a georeferenced GeoTIFF mask for every tile that yields a valid prediction.

    Parameters
    ----------
    imgDF : pd.DataFrame
        Tile index returned by ``doMosaic2tile``.  Must contain columns
        ``mosaic``, ``x_min``, ``y_min``, ``x_max``, ``y_max``.
    my_api_key, my_model_name, my_model_ver : str
        Roboflow credentials and model identifier.
    out_dir : str
        Directory where mask GeoTIFFs will be written.
    minPatchSize : float
        Minimum patch area in CRS units² (e.g. m²) for the small-object filter.
        Pass 0 to skip filtering.
    batch_size, threadCnt : int
        Kept for API compatibility; inference is currently serial (Roboflow SDK
        is not thread-safe with a single model instance).

    Returns
    -------
    pd.DataFrame
        Copy of *imgDF* with an added ``mask_tif`` column pointing to the saved
        GeoTIFF for each tile.  Rows whose inference failed are dropped.
    """
    from tqdm import tqdm
    from pingseg.rf_utils import get_model_online

    model = get_model_online(my_api_key, my_model_name, my_model_ver)
    os.makedirs(out_dir, exist_ok=True)

    # Colour table shared across all tiles (up to 9 classes)
    _COLORMAP = {
        0: (51, 102, 204),
        1: (220, 57, 18),
        2: (255, 153, 0),
        3: (16, 150, 24),
        4: (153, 0, 153),
        5: (0, 153, 198),
        6: (221, 68, 119),
        7: (102, 170, 0),
        8: (184, 46, 46),
    }

    mask_tif_paths = []

    for _, row in tqdm(imgDF.iterrows(), total=len(imgDF), desc='RF inference'):
        tile_path = row['mosaic']

        # ------------------------------------------------------------------
        # 1. Convert tile to a clean 3-band PNG (handles 1-band & 3-band)
        # ------------------------------------------------------------------
        infer_path, is_temp = _prepare_tile_for_inference(tile_path)

        try:
            pred = model.predict(infer_path).json()
        finally:
            if is_temp:
                try:
                    os.remove(infer_path)
                except OSError:
                    pass

        # ------------------------------------------------------------------
        # 2. Extract segmentation mask from response
        # ------------------------------------------------------------------
        found = _find_segmentation_mask(pred)
        if found is None:
            if isinstance(pred, dict):
                keys_preview = {k: type(v).__name__ for k, v in pred.items()}
            else:
                keys_preview = type(pred).__name__
            print(f'Warning: no segmentation mask for {tile_path}. '
                  f'Response keys: {keys_preview}. Skipping.')
            mask_tif_paths.append(None)
            continue

        b64_mask, _ = found
        mask_arr = decode_mask_png_base64(b64_mask)  # 2D uint8 array of class IDs

        # ------------------------------------------------------------------
        # 3. Optionally filter small objects
        # ------------------------------------------------------------------
        if minPatchSize > 0:
            x_min_r = row['x_min']
            x_max_r = row['x_max']
            pix_m = (x_max_r - x_min_r) / mask_arr.shape[1]
            try:
                from pingtile.utils import filterLabel
                mask_arr = filterLabel(mask_arr.astype(np.uint8),
                                       min_size=minPatchSize,
                                       pix_m=pix_m)
            except Exception as _e:
                print(f'Warning: filterLabel failed ({_e}); saving unfiltered mask.')

        # ------------------------------------------------------------------
        # 4. Georeference and save mask GeoTIFF
        # ------------------------------------------------------------------
        x_min = row['x_min']
        y_min = row['y_min']
        x_max = row['x_max']
        y_max = row['y_max']

        with rio.open(tile_path) as src:
            crs = src.crs

        height, width = mask_arr.shape
        transform = rio_from_bounds(x_min, y_min, x_max, y_max, width, height)

        base = os.path.splitext(os.path.basename(tile_path))[0]
        out_path = os.path.join(out_dir, f'{base}_mask.tif')

        with rio.open(
            out_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=mask_arr.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.nodata = 0
            dst.write(mask_arr, 1)
            dst.write_colormap(1, _COLORMAP)

        mask_tif_paths.append(out_path)

    result_df = imgDF.copy()
    result_df['mask_tif'] = mask_tif_paths
    result_df = result_df[result_df['mask_tif'].notna()].reset_index(drop=True)

    return result_df


# Example usage with your JSON blob:
if __name__ == "__main__":
    import json
    example = {
      "segmentation_mask": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAAAAADRE4smAAABFUlEQVR4nO3BMQEAAADCoPVP7WkJoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AYCPAABLVGwWAAAAABJRU5ErkJggg==",
      "class_map": {"0":"background","1":"SAV"},
      "image": {"width":360,"height":360}
    }
    fc = mask_json_to_geojson(example, min_area=1.0, simplify_tolerance=0.2)
    # Write to file
    with open("mask_polygons.geojson", "w") as f:
        json.dump(fc, f)
    print("Wrote mask_polygons.geojson with", len(fc["features"]), "features")
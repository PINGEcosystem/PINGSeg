'''
Copyright (c) 2025 Cameron S. Bodine

Adapted from Segmentation Gym: https://github.com/Doodleverse/segmentation_gym

'''

#########
# Imports
import os
import pandas as pd


import base64
import io
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from PIL import Image
import cv2
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





#=======================================================================
def seg_rf_folder(imgDF: pd.DataFrame,
                  my_api_key: str,
                  my_model_name: str,
                  my_model_ver: str,
                  out_dir: str,
                  batch_size: int=8,
                  threadCnt: int=4):
    
    # get the model
    from pingseg.rf_utils import get_model_online
    model = get_model_online(my_api_key, my_model_name, my_model_ver)

    # Do prediction
    file_paths = imgDF["mosaic"].tolist()

    for file in file_paths:
        pred = model.predict(file).json()

        # Try to find the segmentation_mask in the model response. Many model servers
        # nest the base64 PNG in different fields; search recursively and accept
        # data URLs too.
        found = _find_segmentation_mask(pred)
        if found is None:
            # helpful debug output: show top-level keys/types so user can adapt
            if isinstance(pred, dict):
                keys_preview = {k: type(v).__name__ for k, v in pred.items()}
            else:
                keys_preview = type(pred).__name__
            print(f"Warning: no segmentation mask found for file {file}.\nResponse top-level keys/types: {keys_preview}\nSkipping this file.")
            # Optionally you can uncomment the next line to raise instead of skipping
            # raise ValueError("No 'segmentation_mask' key found in input JSON. Response: {}".format(keys_preview))
            continue

        b64_mask, path = found
        mask_json = {
            "segmentation_mask": b64_mask,
            "class_map": pred.get("class_map", CLASS_MAP_DEFAULT) if isinstance(pred, dict) else CLASS_MAP_DEFAULT,
            "image": pred.get("image", {}) if isinstance(pred, dict) else {}
        }
        result = mask_json_to_geojson(mask_json)

        print('\n\n', result)

        






    return


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
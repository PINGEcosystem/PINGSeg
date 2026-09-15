'''
Copyright (c) 2025 Cameron S. Bodine

PyTorch/Transformers segmentation workflow for local Hugging Face models.
'''

#########
# Imports
import os
import sys
import json
from typing import List, Tuple


def _preload_windows_torch_backend() -> None:
	"""Import torch before other native stacks on Windows to avoid DLL conflicts."""

	if os.name != "nt":
		return

	_prepare_windows_torch_runtime()

	try:
		import torch  # noqa: F401
	except (ImportError, OSError):
		# seg_torch_folder() raises the user-facing error if torch is still unavailable.
		pass


def _prepare_windows_torch_runtime() -> None:
	"""Preflight DLL/OpenMP settings for torch on Windows."""

	if os.name != "nt":
		return

	os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

	torch_lib_dir = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
	if os.path.isdir(torch_lib_dir):
		os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ.get("PATH", "")
		try:
			os.add_dll_directory(torch_lib_dir)
		except Exception:
			pass


_prepare_windows_torch_runtime()
_preload_windows_torch_backend()

import numpy as np
import pandas as pd
from tqdm import tqdm


def _is_int_like(value) -> bool:
	"""Return True when value can be losslessly interpreted as an integer."""

	try:
		int(value)
		return True
	except (TypeError, ValueError):
		return False


def _normalize_label_mappings(config: dict) -> dict:
	"""Normalize Hugging Face label maps, including legacy reversed exports."""

	normalized = dict(config)
	raw_id2label = normalized.get("id2label")
	raw_label2id = normalized.get("label2id")

	id2label = None
	label2id = None

	if isinstance(raw_id2label, dict) and raw_id2label:
		if all(_is_int_like(key) for key in raw_id2label) and all(isinstance(value, str) for value in raw_id2label.values()):
			id2label = {int(key): value for key, value in raw_id2label.items()}
		elif all(isinstance(key, str) for key in raw_id2label) and all(_is_int_like(value) for value in raw_id2label.values()):
			id2label = {int(value): key for key, value in raw_id2label.items()}

	if isinstance(raw_label2id, dict) and raw_label2id:
		if all(isinstance(key, str) for key in raw_label2id) and all(_is_int_like(value) for value in raw_label2id.values()):
			label2id = {key: int(value) for key, value in raw_label2id.items()}
		elif all(_is_int_like(key) for key in raw_label2id) and all(isinstance(value, str) for value in raw_label2id.values()):
			label2id = {value: int(key) for key, value in raw_label2id.items()}

	if id2label is None and label2id is not None:
		id2label = {index: label for label, index in label2id.items()}

	if label2id is None and id2label is not None:
		label2id = {label: index for index, label in id2label.items()}

	if id2label is not None:
		normalized["id2label"] = dict(sorted(id2label.items()))
		normalized["num_labels"] = len(id2label)

	if label2id is not None:
		normalized["label2id"] = label2id

	return normalized


def _get_model_image_size(config: dict) -> Tuple[int, int]:
	"""Resolve model input size as (height, width)."""

	image_size = config.get("image_size", 224)
	if isinstance(image_size, int):
		return image_size, image_size
	if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
		return int(image_size[0]), int(image_size[1])
	return 224, 224


IMAGE_NORM_METHODS = ("standardize", "minmax")


def _normalize_chw(arr: np.ndarray, method: str, valid_mask: 'np.ndarray | None' = None) -> np.ndarray:
	"""Normalize a (C, H, W) float32 array using the requested method.

	'standardize' mirrors the training preprocessing (per-channel
	``(x - mean) / std`` followed by a rescale to [0, 1]). 'minmax' is the
	legacy per-tile min-max rescale.

	When valid_mask (H, W) is provided, statistics (mean/std/min/max) are
	computed only over valid (non-nodata) pixels so that edge/swath padding
	doesn't skew the normalization, matching training tiles which contain no
	padding.
	"""

	if valid_mask is not None and not np.all(valid_mask):
		mask = valid_mask.astype(bool)
		if not np.any(mask):
			mask = None
	else:
		mask = None

	if method == "standardize":
		out = np.empty_like(arr, dtype=np.float32)
		for c in range(arr.shape[0]):
			band = arr[c]
			stat_band = band[mask] if mask is not None else band
			n_pixels = stat_band.size
			std = float(np.nanstd(stat_band))
			std = max(std, 1.0 / np.sqrt(n_pixels))
			mean = float(np.nanmean(stat_band))
			standardized = (band - mean) / std
			stat_standardized = standardized[mask] if mask is not None else standardized
			s_min = float(np.nanmin(stat_standardized))
			s_max = float(np.nanmax(stat_standardized))
			if s_max > s_min:
				out[c] = np.clip((standardized - s_min) / (s_max - s_min), 0.0, 1.0)
			else:
				out[c] = 0.0
		return out

	# Legacy per-tile min-max normalization across all channels together.
	stat_arr = arr[np.broadcast_to(mask, arr.shape)] if mask is not None else arr
	arr_min = float(np.nanmin(stat_arr))
	arr_max = float(np.nanmax(stat_arr))
	if arr_max > arr_min:
		return np.clip((arr - arr_min) / (arr_max - arr_min), 0.0, 1.0)
	return np.zeros_like(arr, dtype=np.float32)


def _load_tile_as_chw(path: str,
					  target_hw: Tuple[int, int],
					  num_channels: int = 3,
					  norm_method: str = "standardize",
					  image_mean=None,
					  image_std=None) -> Tuple[np.ndarray, np.ndarray]:
	"""Read a tile from disk and return (pixels, valid_mask).

	pixels is a float32 array shaped (C, H, W) resized to target_hw. valid_mask
	is a uint8 (H, W) array (1 = real sonar data, 0 = nodata/zero-padding),
	resized to target_hw with nearest-neighbour interpolation.

	image_mean/image_std, when provided, mirror the Hugging Face image
	processor's do_normalize step (``(x - mean) / std`` per channel), applied
	after tile standardization/rescale and resizing, matching training-time
	preprocessing.
	"""

	import rasterio as rio

	with rio.open(path) as src:
		arr = src.read()

	# Ensure channel-first array.
	if arr.ndim == 2:
		arr = np.expand_dims(arr, axis=0)

	# Match requested channel count.
	if arr.shape[0] == 1 and num_channels >= 3:
		arr = np.repeat(arr, 3, axis=0)
	elif arr.shape[0] > num_channels:
		arr = arr[:num_channels, :, :]
	elif arr.shape[0] < num_channels:
		pad = np.repeat(arr[-1:, :, :], num_channels - arr.shape[0], axis=0)
		arr = np.concatenate([arr, pad], axis=0)

	arr = arr.astype(np.float32)

	# doMovWin writes exactly 0 into nodata/edge-padded pixels, so a pixel is
	# considered valid when any band carries a non-zero, non-NaN value.
	valid_mask = np.any(arr != 0, axis=0) & ~np.any(np.isnan(arr), axis=0)

	arr = _normalize_chw(arr, norm_method, valid_mask=valid_mask)

	target_h, target_w = target_hw
	if arr.shape[1] != target_h or arr.shape[2] != target_w:
		import torch
		import torch.nn.functional as F

		t = torch.from_numpy(arr).unsqueeze(0)
		t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
		arr = t.squeeze(0).cpu().numpy()

		vm = torch.from_numpy(valid_mask.astype(np.float32))[None, None, ...]
		vm = F.interpolate(vm, size=(target_h, target_w), mode="nearest")
		valid_mask = vm.squeeze(0).squeeze(0).cpu().numpy() > 0.5

	if image_mean is not None and image_std is not None:
		arr = (arr - image_mean[:, None, None]) / image_std[:, None, None]

	return arr.astype(np.float32), valid_mask.astype(np.uint8)


def _save_npz_softmax(pred_chw: np.ndarray,
					  valid_mask: np.ndarray,
					  input_path: str,
					  out_dir: str) -> str:
	"""Persist one prediction array as compressed npz with keys 'softmax' and 'valid'."""

	base = os.path.splitext(os.path.basename(input_path))[0]
	npz_path = os.path.join(out_dir, f"{base}.npz")
	np.savez_compressed(npz_path, softmax=pred_chw.astype(np.float32), valid=valid_mask.astype(np.uint8))
	return npz_path


def seg_torch_folder(imgDF: pd.DataFrame,
					 modelDir: str,
					 out_dir: str,
					 batch_size: int = 8,
					 threadCnt: int = 4,
					 image_norm_method: str = "standardize"):
	"""
	Run semantic segmentation on image tiles using a local Transformers model.

	The output format mirrors the Segmentation Gym backend:
	each tile gets a .npz file with a 'softmax' array shaped (classes, H, W)
	and a 'valid' array shaped (H, W) flagging real sonar data vs. nodata.

	image_norm_method : {'standardize', 'minmax'}
		'standardize' reproduces the training-time per-image standardization
		(``(x - mean) / std`` then rescaled to [0, 1]). 'minmax' is the legacy
		per-tile min-max rescale.
	"""

	if image_norm_method not in IMAGE_NORM_METHODS:
		raise ValueError(f"image_norm_method must be one of {IMAGE_NORM_METHODS}, got {image_norm_method!r}")

	_prepare_windows_torch_runtime()

	# Imported lazily so non-torch workflows still run without these deps.
	try:
		import torch
		import torch.nn.functional as F
		from transformers import AutoConfig, AutoModelForSemanticSegmentation
	except (ImportError, OSError) as exc:
		python_exe = os.path.normpath(sys.executable)
		raise RuntimeError(
			"Failed to load the torch backend in this Python environment. "
			"On Windows this is often a DLL/OpenMP conflict (e.g., WinError 127 loading torch\\lib\\shm.dll). "
			"Before launching MayaMapper, try setting KMP_DUPLICATE_LIB_OK in the shell that starts Python:\n"
			"  PowerShell: $env:KMP_DUPLICATE_LIB_OK='TRUE'\n"
			"  cmd.exe: set KMP_DUPLICATE_LIB_OK=TRUE\n"
			f"Interpreter: {python_exe}\n"
			"If that is not enough, reinstall a clean CPU build in this environment:\n"
			f"  \"{python_exe}\" -m pip uninstall -y torch torchvision torchaudio\n"
			f"  \"{python_exe}\" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio\n"
			"If using conda, prefer a single channel family for scientific libs to reduce runtime conflicts."
		) from exc

	del threadCnt  # kept for API compatibility

	os.makedirs(out_dir, exist_ok=True)

	config_path = os.path.join(modelDir, "config.json")
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Missing model config: {config_path}")

	with open(config_path, "r", encoding="utf-8") as f:
		model_config_json = json.load(f)
	model_config_json = _normalize_label_mappings(model_config_json)

	target_hw = _get_model_image_size(model_config_json)
	model_channels = int(model_config_json.get("num_channels", 3))

	model_type = model_config_json.get("model_type")
	if not model_type:
		raise ValueError(f"Model config is missing 'model_type': {config_path}")

	# Mirror the training-time Hugging Face image processor's do_normalize
	# step (ImageNet mean/std by default), applied after standardize/rescale.
	image_mean = None
	image_std = None
	preprocessor_config_path = os.path.join(modelDir, "preprocessor_config.json")
	if os.path.exists(preprocessor_config_path):
		with open(preprocessor_config_path, "r", encoding="utf-8") as f:
			preprocessor_config_json = json.load(f)
		if preprocessor_config_json.get("do_normalize", True):
			mean = preprocessor_config_json.get("image_mean")
			std = preprocessor_config_json.get("image_std")
			if mean is not None and std is not None:
				image_mean = np.asarray(mean, dtype=np.float32)
				image_std = np.asarray(std, dtype=np.float32)

	hf_config_kwargs = dict(model_config_json)
	hf_config_kwargs.pop("model_type", None)
	hf_config = AutoConfig.for_model(model_type, **hf_config_kwargs)
	model = AutoModelForSemanticSegmentation.from_pretrained(modelDir, config=hf_config, local_files_only=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	file_paths: List[str] = imgDF["mosaic"].tolist()

	with torch.no_grad():
		for start in tqdm(range(0, len(file_paths), batch_size), desc="Torch segmentation"):
			batch_paths = file_paths[start:start + batch_size]
			batch_loaded = [
				_load_tile_as_chw(
					p,
					target_hw=target_hw,
					num_channels=model_channels,
					norm_method=image_norm_method,
					image_mean=image_mean,
					image_std=image_std,
				)
				for p in batch_paths
			]
			batch_np = [item[0] for item in batch_loaded]
			batch_valid = [item[1] for item in batch_loaded]
			pixel_values = torch.from_numpy(np.stack(batch_np, axis=0)).to(device)

			logits = model(pixel_values=pixel_values).logits
			logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
			probs = torch.softmax(logits, dim=1).cpu().numpy()

			for pred, valid, pth in zip(probs, batch_valid, batch_paths):
				_save_npz_softmax(pred, valid, pth, out_dir)

	return imgDF

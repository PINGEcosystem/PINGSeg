'''
Copyright (c) 2025 Cameron S. Bodine

Roboflow semantic-segmentation mapper.

Tiles a sonar mosaic into non-overlapping windows, runs Roboflow inference on
each tile, and assembles the per-tile mask GeoTIFFs back into a mosaic map.

Because the Roboflow API returns a hard class-ID mask (not per-class softmax
probabilities), predictions cannot be averaged across overlapping windows.
The window stride is therefore forced to equal the window size so that tiles
are strictly non-overlapping.

Supports both 1-band (grayscale) and 3-band (colour) sonar mosaics.
1-band tiles are replicated to 3-band RGB before being sent to the model.
'''

#########
# Imports
import os
import shutil
import time
import gc
from glob import glob
from os import cpu_count

import pandas as pd
import psutil

gc.enable()

from pingtile.mosaic2tile import doMosaic2tile
from pingtile.utils import mosaic_maps, maps2Shp_rf


# ======================================================================
def _print_usage():
    cpu_pct = round(psutil.cpu_percent(0.5), 1)
    ram_pct = round(psutil.virtual_memory()[2], 1)
    ram_gb  = round(psutil.virtual_memory()[3] / 1e9, 1)
    print('\n\nCurrent CPU/RAM Usage:')
    print('________________________')
    print('{:<5s} | {:<5s} | {:<5s}'.format('CPU %', 'RAM %', 'RAM [GB]'))
    print('________________________')
    print('{:<5s} | {:<5s} | {:<5s}'.format(str(cpu_pct), str(ram_pct), str(ram_gb)))
    print('________________________\n\n')


# ======================================================================
def do_work(
        inDir: str,
        outDirTop: str,
        projName: str,
        my_api_key: str,
        my_model_name: str,
        my_model_ver: str,
        class_map: dict,
        mapRast: bool,
        mapShp: bool,
        epsg: int,
        windowSize_m: tuple,
        minArea_percent: float,
        threadCnt: float,
        mosaicFileType: str,
        deleteIntData: bool = True,
        minPatchSize: float = 3,
        smoothShp: bool = False,
        smoothTol_m: float = 0.5,
        target_size: list = None,
):
    """
    Run the full Roboflow segmentation mapping pipeline on a directory of
    sonar mosaics.

    Parameters
    ----------
    inDir : str
        Directory (searched recursively) containing sonar mosaic files.
    outDirTop : str
        Root output directory.  A sub-folder named *projName* will be created.
    projName : str
        Project name used for sub-folder and output file naming.
    my_api_key : str
        Roboflow API key.
    my_model_name : str
        Roboflow project / model name (lower-case slug).
    my_model_ver : str | int
        Roboflow model version number.
    class_map : dict
        Mapping of integer class IDs to class names, e.g.
        ``{"0": "background", "1": "SAV"}``.
        Class 0 is treated as NoData and excluded from the shapefile.
    mapRast : bool
        If True, export the mosaicked prediction as a GeoTIFF raster.
    mapShp : bool
        If True, export the predictions as an ESRI Shapefile.
    epsg : int
        EPSG code for all outputs.
    windowSize_m : tuple
        Window size in metres, e.g. ``(50, 50)``.
    minArea_percent : float
        Minimum fraction of a tile that must contain valid (non-zero) sonar
        data for the tile to be kept (0–1).
    threadCnt : float
        Thread / worker count for tiling (positive int = exact count,
        fraction = proportion of CPU cores, 0 = all cores).
    mosaicFileType : str
        Glob extension for mosaic files, e.g. ``'.tif'``.
    deleteIntData : bool
        Remove intermediate working directories when done.
    minPatchSize : float
        Minimum patch area (CRS units²) for small-object filtering.
    smoothShp : bool
        Apply polygon smoothing to the final shapefile.
    smoothTol_m : float
        Simplification tolerance (CRS units) for polygon smoothing.
    target_size : list | None
        ``[height, width]`` passed to ``doMosaic2tile``.  Defaults to
        ``[512, 512]`` when ``None``.
    """

    _debug = False  # set True to skip directory wipe and keep intermediate files

    start_time = time.time()

    if target_size is None:
        target_size = [512, 512]

    # ------------------------------------------------------------------ #
    # Thread count                                                         #
    # ------------------------------------------------------------------ #
    if threadCnt == 0:
        threadCnt = cpu_count()
    elif threadCnt < 0:
        threadCnt = max(1, cpu_count() + int(threadCnt))
    elif threadCnt < 1:
        threadCnt = max(1, int(cpu_count() * threadCnt))
        if threadCnt % 2 == 1:
            threadCnt -= 1
    if threadCnt > cpu_count():
        print(f'\nWARNING: Requested {threadCnt} threads but only {cpu_count()} '
              f'available; using {cpu_count()} instead.')
        threadCnt = cpu_count()

    # ------------------------------------------------------------------ #
    # Output directories                                                   #
    # ------------------------------------------------------------------ #
    outDir = os.path.join(outDirTop, projName)

    if os.path.exists(outDir) and not _debug:
        shutil.rmtree(outDir)

    os.makedirs(outDir, exist_ok=True)

    # Track directories to clean up
    to_delete = {}

    # ------------------------------------------------------------------ #
    # Discover mosaics                                                     #
    # ------------------------------------------------------------------ #
    mosaics = glob(
        os.path.join(inDir, '**', f'*{mosaicFileType}'),
        recursive=True,
    )

    if not mosaics:
        raise FileNotFoundError(
            f'No mosaic files matching *{mosaicFileType} found under {inDir}'
        )

    print(f'\nFound {len(mosaics)} mosaic(s) to process.')

    # ------------------------------------------------------------------ #
    # 1. Tile mosaics (non-overlapping: stride == window size)            #
    # ------------------------------------------------------------------ #
    print('\n\nTiling mosaics...\n\n')

    outSonDir = os.path.join(outDir, 'images')
    os.makedirs(outSonDir, exist_ok=True)

    # Stride equals window size → strictly non-overlapping tiles.
    # Roboflow returns hard class masks, not softmax; overlapping tiles
    # cannot be averaged, so there is no benefit to overlap.
    window_stride = windowSize_m[0]

    imagesAll = []
    for mosaic in mosaics:
        r = doMosaic2tile(
            inFile=mosaic,
            outDir=outSonDir,
            windowSize=windowSize_m,
            windowStride_m=window_stride,
            outName=projName,
            epsg_out=epsg,
            threadCnt=threadCnt,
            target_size=target_size,
            minArea_percent=minArea_percent,
            save_rgb=True,   # preserve colour for 3-band mosaics
        )
        imagesAll.append(r)

    imagesDF = pd.concat(imagesAll, axis=0, ignore_index=True)

    print(f'Image tiles generated: {len(imagesDF)}')
    print('\nDone!')
    print('Time (s):', round(time.time() - start_time, ndigits=1))
    _print_usage()

    if deleteIntData:
        to_delete['outSonDir'] = [outSonDir]

    # ------------------------------------------------------------------ #
    # 2. Roboflow inference → per-tile mask GeoTIFFs                      #
    # ------------------------------------------------------------------ #
    print('\n\nRunning Roboflow inference on sonar tiles...\n\n')
    seg_start = time.time()

    out_masks = os.path.join(outDir, 'masks')
    os.makedirs(out_masks, exist_ok=True)

    from pingseg.seg_rf import seg_rf_folder

    imagesDF = seg_rf_folder(
        imgDF=imagesDF,
        my_api_key=my_api_key,
        my_model_name=my_model_name,
        my_model_ver=my_model_ver,
        out_dir=out_masks,
        minPatchSize=0,      # per-tile filtering is optional; bulk filtering
                             # happens later in maps2Shp_rf
    )

    if imagesDF.empty:
        raise RuntimeError(
            'Inference returned no valid predictions. '
            'Check your API key, model name/version, and input tiles.'
        )

    print(f'\nInference complete: {len(imagesDF)} tile(s) predicted.')
    print('Time (s):', round(time.time() - seg_start, ndigits=1))
    _print_usage()

    if deleteIntData:
        to_delete['outSonDir'] = [outSonDir]   # images already queued above

    # ------------------------------------------------------------------ #
    # 3. Export raster mosaic                                             #
    # ------------------------------------------------------------------ #
    if mapRast:
        print('\n\nExporting prediction as raster mosaic...\n\n')
        rast_start = time.time()

        mask_files = imagesDF['mask_tif'].tolist()
        out_mosaic = os.path.join(outDir, 'mosaic')
        os.makedirs(out_mosaic, exist_ok=True)

        mosaic_maps(mask_files, out_mosaic, projName)

        print('\nDone!')
        print('Time (s):', round(time.time() - rast_start, ndigits=1))
        _print_usage()

    # ------------------------------------------------------------------ #
    # 4. Export shapefile                                                  #
    # ------------------------------------------------------------------ #
    if mapShp:
        print('\n\nExporting prediction as shapefile...\n\n')
        shp_start = time.time()

        mask_files = imagesDF['mask_tif'].tolist()
        out_shp = os.path.join(outDir, 'map_shp')
        os.makedirs(out_shp, exist_ok=True)

        maps2Shp_rf(
            map_files=mask_files,
            outDir=out_shp,
            outName=projName,
            class_map=class_map,
            minPatchSize=minPatchSize,
            smoothShp=smoothShp,
            smoothTol_m=smoothTol_m,
        )

        print('\nDone!')
        print('Time (s):', round(time.time() - shp_start, ndigits=1))
        _print_usage()

    # ------------------------------------------------------------------ #
    # 5. Cleanup intermediate data                                        #
    # ------------------------------------------------------------------ #
    gc.collect()

    if deleteIntData:
        print('\n\nDeleting intermediate data...\n\n')

        for name, paths in to_delete.items():
            for path in paths:
                print(f'  {name} -> {path}')
                shutil.rmtree(path, ignore_errors=True)

        # Always remove per-tile masks after export (they sum to hundreds of
        # individual GeoTIFFs)
        shutil.rmtree(out_masks, ignore_errors=True)

    print('\n\nAll done!')
    print('Total time (s):', round(time.time() - start_time, ndigits=1))
    _print_usage()

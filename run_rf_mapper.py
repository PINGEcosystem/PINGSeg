'''
Copyright (c) 2025 Cameron S. Bodine

Run the Roboflow semantic-segmentation mapper (rf_mapper.do_work).

Edit the parameters in the USER PARAMETERS section below, then run:

    conda activate ping
    python run_rf_mapper.py
'''

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pingseg.rf_mapper import do_work

# ==============================================================================
# USER PARAMETERS
# ==============================================================================

# --- Inputs -------------------------------------------------------------------
inDir           = r'C:\path\to\sonar\mosaics'   # directory of mosaic files
mosaicFileType  = '.tif'                         # mosaic file extension

# --- Outputs ------------------------------------------------------------------
outDirTop       = r'C:\path\to\output'           # root output directory
projName        = 'my_project'                   # sub-folder and file name prefix

# --- Roboflow -----------------------------------------------------------------
my_api_key      = 'YOUR_ROBOFLOW_API_KEY'
my_model_name   = 'your-model-slug'              # lower-case project slug
my_model_ver    = '1'                            # model version number

# Class map: integer pixel value -> class name.
# Class 0 is treated as NoData and excluded from the shapefile.
class_map = {
    '0': 'background',
    '1': 'SAV',
}

# --- Projection ---------------------------------------------------------------
epsg            = 32618                          # EPSG code for all outputs

# --- Tiling -------------------------------------------------------------------
windowSize_m    = (50, 50)     # window size in metres (height, width)
minArea_percent = 0.25         # min fraction of tile with valid sonar data (0-1)
target_size     = [512, 512]   # tile size in pixels sent to the model [H, W]

# --- Inference / mapping ------------------------------------------------------
mapRast         = True         # export mosaicked raster prediction
mapShp          = True         # export prediction as shapefile
minPatchSize    = 3.0          # min polygon area (CRS units²) kept in shapefile
smoothShp       = False        # simplify polygon boundaries
smoothTol_m     = 0.5          # simplification tolerance (CRS units)

# --- Resources ----------------------------------------------------------------
# Positive int  = exact thread count
# Float 0-1     = fraction of available cores
# 0             = all cores
threadCnt       = 4

# Keep intermediate tile images and mask GeoTIFFs when False (useful for
# debugging).
deleteIntData   = True

# ==============================================================================
# RUN
# ==============================================================================

if __name__ == '__main__':
    do_work(
        inDir           = inDir,
        outDirTop       = outDirTop,
        projName        = projName,
        my_api_key      = my_api_key,
        my_model_name   = my_model_name,
        my_model_ver    = my_model_ver,
        class_map       = class_map,
        mapRast         = mapRast,
        mapShp          = mapShp,
        epsg            = epsg,
        windowSize_m    = windowSize_m,
        minArea_percent = minArea_percent,
        threadCnt       = threadCnt,
        mosaicFileType  = mosaicFileType,
        deleteIntData   = deleteIntData,
        minPatchSize    = minPatchSize,
        smoothShp       = smoothShp,
        smoothTol_m     = smoothTol_m,
        target_size     = target_size,
    )

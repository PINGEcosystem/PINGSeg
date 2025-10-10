'''
Copyright (c) 2025 Cameron S. Bodine

Adapted from Segmentation Gym: https://github.com/Doodleverse/segmentation_gym

'''

#########
# Imports
import os, sys
from glob import glob
import pandas as pd
import json
import rasterio as rio
import numpy as np
import tensorflow as tf
from imageio import imread

from doodleverse_utils.imports import *
from doodleverse_utils.model_imports import *
from doodleverse_utils.prediction_imports import *

#=======================================================================
def get_model(modelDir: str,
              MODEL: str,
              TARGET_SIZE: list,
              N_DATA_BANDS: int,
              FILTERS: int,
              NCLASSES: int,
              KERNEL: int,
              STRIDE: int,
              DROPOUT: float,
              DROPOUT_CHANGE_PER_LAYER: float,
              DROPOUT_TYPE: str,
              USE_DROPOUT_ON_UPSAMPLING: bool,
              ):

    '''
    '''

    if MODEL =='resunet':
        model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                        )
    elif MODEL=='unet':
        model =  custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                        )

    elif MODEL =='simple_resunet':

        model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    elif MODEL=='simple_unet':
        model = simple_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    elif MODEL=='satunet':

        model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    filters=FILTERS,
                    num_layers=4,
                    strides=(1,1))

    elif MODEL=='segformer':
        id2label = {}
        for k in range(NCLASSES):
            id2label[k]=str(k)
        model = segformer(id2label,num_classes=NCLASSES)
        # model.compile(optimizer='adam')

    else:
        print("Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'")
        sys.exit(2)

    return model
              
#=======================================================================
def seg_file2tensor_3band(f, TARGET_SIZE):  
    """
    "seg_file2tensor(f)"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """

    if isinstance(f, bytes):
        f = f.decode()

    bigimage = imread(f)  
    smallimage = resize(
        bigimage, (TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True
    )
    smallimage = np.array(smallimage)
    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage

#=======================================================================
def seg_file2tensor_ND(f, TARGET_SIZE):  
    """
    "seg_file2tensor(f)"
    This function reads a NPZ image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of npz
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """

    with np.load(f) as data:
        bigimage = data["arr_0"].astype("uint8")

    smallimage = resize(
        bigimage, (TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True
    )
    smallimage = np.array(smallimage)
    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage

#=======================================================================
def get_image(f: str, 
              N_DATA_BANDS: int, 
              TARGET_SIZE: list, 
              MODEL: str):

    if N_DATA_BANDS <= 3:
        image, w, h, bigimage = seg_file2tensor_3band(f, TARGET_SIZE)
    else:
        image, w, h, bigimage = seg_file2tensor_ND(f, TARGET_SIZE)

    try: ##>3 bands
        if N_DATA_BANDS<=3:
            if image.shape[-1]>3:
                image = image[:,:,:3]

            if bigimage.shape[-1]>3:
                bigimage = bigimage[:,:,:3]
    except:
        pass

    image = standardize(image.numpy()).squeeze()

    if isinstance(MODEL, bytes):
        MODEL = MODEL.decode()

    if str(MODEL) == 'segformer':
        if np.ndim(image) == 2:
            image = np.dstack((image, image, image))
        # Ensure channels first (3, H, W)
        if image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))

    image = image.astype(np.float32)

    return image#, w, h, bigimage


#=======================================================================
def seg_gym_folder(imgDF: pd.DataFrame,
                   modelDir: str,
                   out_dir: str,
                   batch_size: int=8
                   ):
    '''
    '''

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get necessary model files
    configFile = glob(os.path.join(modelDir, 'config', '*.json'))[0]
    weights = glob(os.path.join(modelDir, 'weights', '*_fullmodel.h5'))[0]

    # Get info from configfile
    with open(configFile) as f:
        config = json.load(f)

    model = get_model(modelDir=modelDir,
                      MODEL=config['MODEL'],
                      TARGET_SIZE=config['TARGET_SIZE'],
                      N_DATA_BANDS=config['N_DATA_BANDS'],
                      FILTERS=config['FILTERS'],
                      NCLASSES=config['NCLASSES'],
                      KERNEL=config['KERNEL'],
                      STRIDE=config['STRIDE'],
                      DROPOUT=config['DROPOUT'],
                      DROPOUT_CHANGE_PER_LAYER=config['DROPOUT_CHANGE_PER_LAYER'],
                      DROPOUT_TYPE=config['DROPOUT_TYPE'],
                      USE_DROPOUT_ON_UPSAMPLING=config['USE_DROPOUT_ON_UPSAMPLING']
                      )
    
    # Load the weights
    model.load_weights(weights)
    
    file_paths = imgDF["mosaic"].tolist()
    target_size = config['TARGET_SIZE']
    n_data_bands = config['N_DATA_BANDS']

    if n_data_bands < 3 and config['MODEL'] == 'segformer':
        n_data_bands = 3  # Segformer expects 3 bands for RGB input

    # Data Loader
    def tf_load_img(path):
        img = tf.numpy_function(
            func=get_image,
            inp=[path, n_data_bands, target_size, config['MODEL']],
            Tout=tf.float32
        )
        img.set_shape([n_data_bands, target_size[0], target_size[1]])
        return img

    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(lambda x: tf_load_img(x), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    for i, (batch, batch_paths) in enumerate(zip(ds, [file_paths[j:j+batch_size] for j in range(0, len(file_paths), batch_size)])):
        preds = model.predict(batch, verbose=1)

        # print('\n\n\n')
        # print_attrs(preds)

        # Save softmax scores for each image in the batch
        for pred, path in zip(preds['logits'], batch_paths):
            base = os.path.splitext(os.path.basename(path))[0]
            npz_path = os.path.join(out_dir, f"{base}.npz")
            np.savez_compressed(npz_path, softmax=pred)

    return imgDF




#=======================================================================
def seg_gym_folder_noDL(imgDF: pd.DataFrame,
                   modelDir: str,
                   out_dir: str,
                   batch_size: int=8
                   ):
    '''
    do segmentation without a dataloader
    '''

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get necessary model files
    configFile = glob(os.path.join(modelDir, 'config', '*.json'))[0]
    weights = glob(os.path.join(modelDir, 'weights', '*_fullmodel.h5'))[0]

    # Get info from configfile
    with open(configFile) as f:
        config = json.load(f)

    model = get_model(modelDir=modelDir,
                      MODEL=config['MODEL'],
                      TARGET_SIZE=config['TARGET_SIZE'],
                      N_DATA_BANDS=config['N_DATA_BANDS'],
                      FILTERS=config['FILTERS'],
                      NCLASSES=config['NCLASSES'],
                      KERNEL=config['KERNEL'],
                      STRIDE=config['STRIDE'],
                      DROPOUT=config['DROPOUT'],
                      DROPOUT_CHANGE_PER_LAYER=config['DROPOUT_CHANGE_PER_LAYER'],
                      DROPOUT_TYPE=config['DROPOUT_TYPE'],
                      USE_DROPOUT_ON_UPSAMPLING=config['USE_DROPOUT_ON_UPSAMPLING']
                      )
    
    # Load the weights
    model.load_weights(weights)
    
    file_paths = imgDF["mosaic"].tolist()
    target_size = config['TARGET_SIZE']
    n_data_bands = config['N_DATA_BANDS']

    if n_data_bands < 3 and config['MODEL'] == 'segformer':
        n_data_bands = 3  # Segformer expects 3 bands for RGB input

    for im_path in file_paths:
        # Get image basename
        basename = os.path.splitext(os.path.basename(im_path))[0]

        # Load image directly (avoid tf.numpy_function here because that
        # can wrap the path in an array-like and pass that to imread).
        image = get_image(im_path, n_data_bands, target_size, config['MODEL'])

        # Ensure batch dimension for model.predict: shape should be (1, ...)
        try:
            import numpy as _np
            batch = _np.expand_dims(image, 0)
        except Exception:
            batch = image

        preds = model.predict(batch, verbose=0)

        # preds may be a dict (e.g., {'logits': array}) or an array with batch dim.
        if isinstance(preds, dict):
            # Extract logits/softmax and remove batch dim
            arr = preds['logits']
        else:
            arr = preds

        # If arr has a batch dimension, take first element
        try:
            arr0 = arr[0]
        except Exception:
            arr0 = arr

        npz_path = os.path.join(out_dir, f"{basename}.npz")
        np.savez_compressed(npz_path, softmax=arr0)

    return(imgDF)




# For Debug
import reprlib, inspect
def short_repr(x, maxlen=200):
    try:
        r = repr(x)
    except Exception:
        return f"<unrepr-able {type(x).__name__}>"
    if len(r) > maxlen:
        return r[:maxlen] + '...'
    return r

def print_attrs(obj, show_private=False, maxlen=200):
    print("Type:", type(obj))
    # If object has __dict__ show those first
    d = getattr(obj, '__dict__', None)
    if d:
        print("__dict__:")
        for k, v in d.items():
            if not show_private and k.startswith('_'):
                continue
            print(f"  {k} ({type(v).__name__}): {short_repr(v, maxlen)}")
        return

    # fallback: inspect.getmembers
    for name, val in inspect.getmembers(obj):
        if not show_private and name.startswith('_'):
            continue
        # skip bound methods if you want only attributes
        if inspect.ismethod(val) or inspect.isfunction(val):
            print(f"  {name}()  # method")
        else:
            print(f"  {name} ({type(val).__name__}): {short_repr(val, maxlen)}")
# for debug
              
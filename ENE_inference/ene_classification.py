import os
import datetime
import argparse
import logging
import sys
from scipy import ndimage
from scipy.ndimage import binary_dilation
import numpy as np
import pandas as pd
import random
from crop_roi import crop_roi_ene
from util import bbox2_3D
import feret
import SimpleITK as sitk




# Resolve repository-relative paths (assumes this script lives in ENE_inference/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, 'ene_model', '0208-1531-1_DualNet.h5')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_dir')

# Optional GPU selection via CLI (default: auto)
parser = argparse.ArgumentParser(description="pENE inference pipeline")
parser.add_argument("--gpu", default="auto", help="GPU index (e.g., 0), 'cpu', or 'auto' [default]")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
parser.add_argument("--seed", type=int, default=42, help="Random seed for numpy/python/tensorflow")
parser.add_argument("--seg-dir", dest="seg_dir", default="./data/seg", help="Path to segmentation directory")
parser.add_argument("--img-dir", dest="img_dir", default="./data/img", help="Path to image directory")
parser.add_argument("--output-dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR, help="Base output directory (default: ENE_inference/output_dir)")
parser.add_argument("--limit", type=int, default=None, help="Max number of cases to process (default: all)")
parser.add_argument("--include-short-axes", action="store_true", default=False, help="Include short/long axis metrics in CSV output")
parser.add_argument("--fg-labels", default="2", help="Foreground label value(s) in segmentation (comma-separated) or 'any' for any nonzero")
parser.add_argument("--norm", default="legacy", choices=["legacy", "zscore", "none"], help="Intensity normalization: legacy (+0.001 & clip), zscore, or none")
args, _ = parser.parse_known_args()

if isinstance(args.gpu, str) and args.gpu.lower() == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif isinstance(args.gpu, str) and args.gpu.lower() == "auto":
    pass  # do not set; let TF use available GPUs
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)



# Import TensorFlow/Keras only after device visibility is set
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# If explicitly on CPU, also hide GPUs at TF level
try:
    if isinstance(args.gpu, str) and args.gpu.lower() == "cpu":
        tf.config.set_visible_devices([], 'GPU')
except Exception as e:
    logging.warning(f"Failed to set TF visible devices to CPU-only: {e}")

# Configure logging to console
logging.basicConfig(
    level=logging.INFO if args.verbose else logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Seeding
try:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    logging.info(f"Seeding set (seed={args.seed})")
except Exception as e:
    logging.warning(f"Failed to set seeds/determinism: {e}")

# Configure TF memory growth (if GPU present)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"GPUs detected: {[d.name for d in physical_devices]}")
    except RuntimeError as e:
        logging.warning(str(e))
else:
    logging.info('Using CPU')


# Parameters
# num_epoch = 20
# batch_size = 10
# limit_num = None





# Expect utils to be installed as a package (pip install -e ENE_inference/data-utils)

def zscore_nonzero(x):
    nz = x[x != 0]
    m = float(nz.mean()) if nz.size else 0.0
    s = float(nz.std()) if nz.size else 1.0
    if s < 1e-6:
        s = 1.0
    y = (x - m) / s
    return np.clip(y, -5, 5)

def normalize_legacy(x):
    y = x + 0.001
    y[y > 1] = 1
    return y

def orient_arr_box_to_zyx(arr, expected=(32, 118, 118)):
    if arr.ndim != 3:
        raise ValueError(f"arr_box must be 3D, got {arr.ndim}D")
    if arr.shape == expected:
        return arr
    target_z = expected[0]
    if target_z in arr.shape:
        z_axis = arr.shape.index(target_z)
        order = (z_axis,) + tuple(i for i in range(3) if i != z_axis)
        arr = np.transpose(arr, order)
        # If the two spatial dims are swapped, fix them
        if arr.shape[0] == expected[0] and set(arr.shape[1:]) == set(expected[1:]):
            if arr.shape[1:] != expected[1:]:
                arr = np.transpose(arr, (0, 2, 1))
        if arr.shape == expected:
            return arr
    raise ValueError(f"arr_box has shape {arr.shape}; expected {expected} (after orientation)")

def resample_to_isotropic(image_sitk, mask_array, spacing_out=(1.0, 1.0, 1.0)):
    spacing_in = image_sitk.GetSpacing()
    size_in    = image_sitk.GetSize()
    origin     = image_sitk.GetOrigin()
    direction  = image_sitk.GetDirection()

    size_out = [int(round(size_in[i] * spacing_in[i] / spacing_out[i])) for i in range(3)]

    mask_sitk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask_sitk.SetOrigin(origin); mask_sitk.SetSpacing(spacing_in); mask_sitk.SetDirection(direction)

    identity = sitk.Transform()

    image_iso = sitk.Resample(
        image_sitk, size_out, identity, sitk.sitkLinear,
        origin, spacing_out, direction, 0.0, sitk.sitkFloat32
    )
    mask_iso = sitk.Resample(
        mask_sitk, size_out, identity, sitk.sitkNearestNeighbor,
        origin, spacing_out, direction, 0, sitk.sitkUInt8
    )
    return image_iso, mask_iso


# (GPU selection handled via CLI above)

### LOAD ENE MODEL ###
K.clear_session()

MODEL_NAME = "0208-1531-1_DualNet_multi.h5"
model = load_model(DEFAULT_MODEL_PATH)

segmentation_folder = args.seg_dir
image_path = args.img_dir

# Add timestamp to create unique directory names
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Use CLI-provided output directory
base_output_dir = args.output_dir
output_folder = os.path.join(base_output_dir, f"output_{timestamp}")
label_path = os.path.join(base_output_dir, f"label_croptop_172x172x76_{timestamp}")
prediction_folder = os.path.join(os.path.join(base_output_dir, "prediction_list"), f"predictions_{timestamp}")

try:
    os.makedirs(label_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
except Exception as e:
    logging.warning(f"Could not create output directories: {e}")

# Add file handler to log into the output folder
log_path = os.path.join(output_folder, f"run_{timestamp}.log")
try:
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO if args.verbose else logging.WARNING)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(fh)
    logging.info("Label and output paths ready")
except Exception as e:
    logging.warning(f"Failed to attach file logger: {e}")

image_type = ".nii"
roi_size = (172, 172, 76)
prediction_list = []
processed_count = 0
limit = args.limit  # None means no limit
for file in sorted(os.listdir(segmentation_folder)):
    if image_type in file:
        database = "bwh"
        case_id = file.split('.')[0]
        logging.info(f"STARTING: case_id: {case_id}")
        path_label = os.path.join(segmentation_folder, file)
        logging.info(f"Processing: {file}")
        processed_count += 1
        if limit is not None and processed_count > limit:
            break
        sitk_label = sitk.ReadImage(path_label, sitk.sitkUInt8)
        np_label = sitk.GetArrayFromImage(sitk_label)
        # Track presence of label 1 (e.g., malignant / positive node without ENE)
        has_label_1 = np.any(np_label == 1)

        # Standardize labels: accept any nonzero or specific label set
        fg_arg = str(getattr(args, "fg_labels", "2")).strip().lower()
        if fg_arg == "any":
            bin_mask = (np_label != 0)
            logging.info("Using any nonzero voxel as foreground for case %s", case_id)
        else:
            try:
                fg_vals = {int(v) for v in fg_arg.split(',') if v.strip() != ''}
            except ValueError:
                fg_vals = {2}
            bin_mask = np.isin(np_label, list(fg_vals))
            logging.info("Using foreground labels %s for case %s", sorted(list(fg_vals)), case_id)

        # If there is no foreground according to fg-labels, optionally emit a
        # dummy row when label 1 is present, to indicate "malignant node but no ENE node"
        if not np.any(bin_mask):
            if has_label_1:
                logging.info(
                    "Case %s: no LN foreground (fg-labels=%s) but label 1 present; "
                    "adding dummy classification row with zeros.", case_id, fg_arg
                )
                if 'multi' in MODEL_NAME:
                    prediction_list.append([
                        case_id,
                        -1,   # synthetic node id
                        0.0,  # pos
                        0.0,  # ene
                        0.0,  # tumor_volume_after_dilation
                        0.0,  # tumor_volume_before_dilation
                        0.0,  # short_axis_axial
                        0.0,  # short_axis_coronal
                        0.0,  # short_axis_sagittal
                        0.0,  # long_axis_axial
                        0.0,  # long_axis_coronal
                        0.0,  # long_axis_sagittal
                        0.0,  # largest_short_axis
                    ])
                elif 'sigmoid' in MODEL_NAME and 'multi' not in MODEL_NAME:
                    prediction_list.append([
                        case_id,
                        -1,   # synthetic node id
                        0.0,  # not_ene
                        0.0,  # ene
                        0.0,  # tumor_volume_after_dilation
                        0.0,  # tumor_volume_before_dilation
                        0.0,  # short_axis_axial
                        0.0,  # short_axis_coronal
                        0.0,  # short_axis_sagittal
                        0.0,  # long_axis_axial
                        0.0,  # long_axis_coronal
                        0.0,  # long_axis_sagittal
                        0.0,  # largest_short_axis
                    ])
                else:  # softmax fallback
                    prediction_list.append([
                        case_id,
                        -1,   # synthetic node id
                        0.0,  # pos
                        0.0,  # ene
                        0.0,  # tumor_volume_after_dilation
                        0.0,  # tumor_volume_before_dilation
                        0.0,  # short_axis_axial
                        0.0,  # short_axis_coronal
                        0.0,  # short_axis_sagittal
                        0.0,  # long_axis_axial
                        0.0,  # long_axis_coronal
                        0.0,  # long_axis_sagittal
                        0.0,  # largest_short_axis
                    ])
            else:
                logging.info(
                    "Case %s: no foreground voxels for fg-labels=%s and no label 1 present; "
                    "skipping case.", case_id, fg_arg
                )
            # Skip further processing for this case
            continue

        np_label = bin_mask.astype(np.uint8)
        new_label = sitk.GetImageFromArray(np_label)
        new_label.CopyInformation(sitk_label)
        sitk_label = new_label

        ### Splitting Nodes / Targets ###
        dist_img = sitk.SignedMaurerDistanceMap(sitk_label != 0, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)
        radius = 2
        seeds = sitk.ConnectedComponent(dist_img < radius)
        seeds = sitk.RelabelComponent(seeds, minimumObjectSize=100) 
        ws = sitk.MorphologicalWatershed(dist_img, markWatershedLine=False, level=1)
        interpolated_label = sitk.Mask(ws, sitk.Cast(seeds, ws.GetPixelID()))
        sitk.WriteImage(interpolated_label, os.path.join(label_path, case_id + "_label_split.nrrd"))
        
        ### Pull in Cropped Image ###
        image = sitk.ReadImage(os.path.join(image_path, case_id + '_0000.nii.gz'))

        interpolated_label = sitk.GetArrayFromImage(interpolated_label)
        node_vol = {}
        for node in np.unique(interpolated_label)[1:]:
            array_mask = np.zeros(interpolated_label.shape)
            array_mask = np.where(interpolated_label == node, 1, 0)
            node_vol[node] = np.sum(array_mask)

        nodes_big = node_vol
        for node in nodes_big:
            array_mask = np.zeros(interpolated_label.shape)
            array_mask = np.where(interpolated_label == node, 1, 0)



            ### Calculate the tumor volume BEFORE dilation ###
            voxel_spacing = image.GetSpacing()  # Get voxel spacing in (x, y, z)
            voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]  # volume of one voxel (in mm³)
            tumor_voxels_before_dilation = np.count_nonzero(array_mask)  # Count the number of tumor voxels
            tumor_volume_before_dilation = tumor_voxels_before_dilation * voxel_volume  # Calculate the tumor volume



            ### Calculate the Feret diameters BEFORE dilation (optional) ###
            if getattr(args, "include_short_axes", False):
                # Resample image and mask to isotropic spacing using SimpleITK
                isotropic_image_sitk, isotropic_mask_sitk = resample_to_isotropic(image, array_mask, spacing_out=(1.0,1.0,1.0))
                isotropic_array_mask = sitk.GetArrayFromImage(isotropic_mask_sitk)

                # Axial plane (slices orthogonal to z-axis)
                max_area = 0
                best_slice_axial = 0
                for z in range(isotropic_array_mask.shape[0]):
                    slice_mask = isotropic_array_mask[z, :, :].astype(np.uint8)
                    area = np.sum(slice_mask)
                    if area > max_area:
                        max_area = area
                        best_slice_axial = z

                slice_mask = isotropic_array_mask[best_slice_axial, :, :].astype(np.uint8)
                if np.sum(slice_mask) > 0:
                    res = feret.calc(slice_mask)
                    maxf_axial = res.maxf
                    maxf90_axial = res.maxf90
                    short_axis_axial = maxf90_axial
                    long_axis_axial = maxf_axial
                else:
                    short_axis_axial = 0
                    long_axis_axial = 0

                # Coronal plane
                max_area = 0
                best_slice_coronal = 0
                for y in range(isotropic_array_mask.shape[1]):
                    slice_mask = isotropic_array_mask[:, y, :].astype(np.uint8)
                    area = np.sum(slice_mask)
                    if area > max_area:
                        max_area = area
                        best_slice_coronal = y

                slice_mask = isotropic_array_mask[:, best_slice_coronal, :].astype(np.uint8)
                if np.sum(slice_mask) > 0:
                    res = feret.calc(slice_mask)
                    maxf_coronal = res.maxf
                    maxf90_coronal = res.maxf90
                    short_axis_coronal = maxf90_coronal
                    long_axis_coronal = maxf_coronal
                else:
                    short_axis_coronal = 0
                    long_axis_coronal = 0

                # Sagittal plane
                max_area = 0
                best_slice_sagittal = 0
                for x in range(isotropic_array_mask.shape[2]):
                    slice_mask = isotropic_array_mask[:, :, x].astype(np.uint8)
                    area = np.sum(slice_mask)
                    if area > max_area:
                        max_area = area
                        best_slice_sagittal = x

                slice_mask = isotropic_array_mask[:, :, best_slice_sagittal].astype(np.uint8)
                if np.sum(slice_mask) > 0:
                    res = feret.calc(slice_mask)
                    maxf_sagittal = res.maxf
                    maxf90_sagittal = res.maxf90
                    short_axis_sagittal = maxf90_sagittal
                    long_axis_sagittal = maxf_sagittal
                else:
                    short_axis_sagittal = 0
                    long_axis_sagittal = 0

                short_axes = [short_axis_axial, short_axis_coronal, short_axis_sagittal]
                short_axes = [axis for axis in short_axes if axis > 0]
                largest_short_axis = max(short_axes) if short_axes else 0
            else:
                # Skip short-axis computation when not requested
                short_axis_axial = 0
                short_axis_coronal = 0
                short_axis_sagittal = 0
                long_axis_axial = 0
                long_axis_coronal = 0
                long_axis_sagittal = 0
                largest_short_axis = 0



            ### Apply dilation AFTER calculations ###
            array_mask = binary_dilation(
                array_mask.astype(bool),
                structure=np.ones((1,5,5), np.uint8),
                iterations=2
            ).astype(np.uint8)



            ### Calculate the tumor volume after dilation ###
            voxel_spacing = image.GetSpacing()  # Get voxel spacing in (x, y, z)
            voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]  # volume of one voxel (in mm³)
            tumor_voxels = np.count_nonzero(array_mask)  # Count the number of tumor voxels
            tumor_volume = tumor_voxels * voxel_volume  # Calculate the tumor volume


            ### Save Mask ###
            mask = sitk.GetImageFromArray(array_mask)
            mask.SetSpacing(image.GetSpacing())
            mask.SetOrigin(image.GetOrigin())
            mask.SetDirection(image.GetDirection())
            if not os.path.exists(os.path.join(label_path, "split_nodes")):
                os.makedirs(os.path.join(label_path, "split_nodes"))
            sitk.WriteImage(mask, os.path.join(label_path, "split_nodes", f"{case_id}_{node}_label.nrrd"))

            ### Center and Crop to BOX NET ###
            path_split_nodes = os.path.join(label_path, "split_nodes")
            crop_shape = (118, 118, 32)
            return_type = 'numpy_array'
            label_id = f"{case_id}_{node}"
            path_to_label_nrrd = os.path.join(path_split_nodes, f"{label_id}_label.nrrd")
            path_to_image_nrrd = os.path.join(image_path, f"{case_id}_0000.nii.gz")
            output_folder_image = os.path.join(output_folder, 'image_crop')
            output_folder_label = os.path.join(output_folder, 'label_crop')
            if not os.path.exists(output_folder_image):
                os.makedirs(output_folder_image)
            if not os.path.exists(output_folder_label):
                os.makedirs(output_folder_label)
            image_obj, label_obj = crop_roi_ene(database,
                                            label_id,
                                            label_id,
                                            path_to_image_nrrd, 
                                            path_to_label_nrrd, 
                                            crop_shape,
                                            return_type, 
                                            output_folder_image, 
                                            output_folder_label)
            
            arr_box = np.multiply(image_obj, label_obj)
            rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(arr_box)
            cropped_image = image_obj[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]
            cropped_mask = label_obj[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1]
            sizefactor = [32/x for x in cropped_image.shape]
            resized_image = ndimage.zoom(cropped_image, zoom=sizefactor, mode='constant', order=1)
            resized_mask = ndimage.zoom(cropped_mask, zoom=sizefactor, mode='nearest', order=0)
            resized_mask = np.round(resized_mask)
            smallinput = np.multiply(resized_image, resized_mask)
            smallinput.shape  

            # Normalize intensities based on selected strategy
            if getattr(args, "norm", "legacy") == "zscore":
                logging.info("Using zscore normalization")
                arr_box = zscore_nonzero(arr_box)
                smallinput = zscore_nonzero(smallinput)
            elif getattr(args, "norm", "legacy") == "legacy":
                logging.info("Using legacy normalization (+0.001 then clip to 1)")
                arr_box = normalize_legacy(arr_box)
                smallinput = normalize_legacy(smallinput)
            else:
                logging.info("Using no normalization")

            img_box = sitk.GetImageFromArray(arr_box)
            sitk.WriteImage(img_box, os.path.join(output_folder, "image_crop", f"{case_id}_{node}_box_input.nrrd"))
            
            imageinput_small = sitk.GetImageFromArray(smallinput)
            sitk.WriteImage(imageinput_small, os.path.join(output_folder, "image_crop", f"{case_id}_{node}_small_input.nrrd"))       

            ## TESTING ON ENE MODEL ##        
            # Validate and correct axis order before reshape
            expected_main = (32, 118, 118)
            expected_small = (32, 32, 32)

            arr_box = orient_arr_box_to_zyx(arr_box, expected_main)

            if smallinput.shape != expected_small:
                raise ValueError(f"smallinput has shape {smallinput.shape}; expected {expected_small}")

            x_testS = smallinput.reshape(1, 32, 32, 32, 1)
            x_test = arr_box.reshape(1, 32, 118, 118, 1)
            batch_size = 24
            predict = model.predict([x_test, x_testS], batch_size) 
            if 'multi' in MODEL_NAME:
                logging.info(f"case_id: {case_id}, Node #: {node}, pos: {predict[0,0]}, ENE: {predict[0,1]}, Tumor Volume: {tumor_volume}, Largest Short Axis (mm): {largest_short_axis}")
                prediction_list.append([
                    case_id,
                    node,
                    predict[0, 0],
                    predict[0, 1],
                    tumor_volume,
                    tumor_volume_before_dilation,
                    short_axis_axial,
                    short_axis_coronal,
                    short_axis_sagittal,
                    long_axis_axial,
                    long_axis_coronal,
                    long_axis_sagittal,
                    largest_short_axis,
                ])
                df_pred = pd.DataFrame(prediction_list, columns=[
                    'case_id',
                    'node',
                    'pos',
                    'ene',
                    'tumor_volume_after_dilation',
                    "tumor_volume_before_dilation",
                    'short_axis_axial',
                    'short_axis_coronal',
                    'short_axis_sagittal',
                    'long_axis_axial',
                    'long_axis_coronal',
                    'long_axis_sagittal',
                    'largest_short_axis',
                ])
            elif 'sigmoid' in MODEL_NAME and 'multi' not in MODEL_NAME:
                logging.info(f"case_id: {case_id}, Node #: {node}, pos: {1 - predict[0,0]}, ENE: {predict[0,0]}, Tumor Volume: {tumor_volume}, Largest Short Axis (mm): {largest_short_axis}")
                prediction_list.append([
                    case_id,
                    node,
                    1 - predict[0, 0],
                    predict[0, 0],
                    tumor_volume,
                    tumor_volume_before_dilation,
                    short_axis_axial,
                    short_axis_coronal,
                    short_axis_sagittal,
                    long_axis_axial,
                    long_axis_coronal,
                    long_axis_sagittal,
                    largest_short_axis,
                ])
                df_pred = pd.DataFrame(prediction_list, columns=[
                    'case_id',
                    'node',
                    'not_ene',
                    'ene',
                    'tumor_volume_after_dilation',
                    "tumor_volume_before_dilation",
                    'short_axis_axial',
                    'short_axis_coronal',
                    'short_axis_sagittal',
                    'long_axis_axial',
                    'long_axis_coronal',
                    'long_axis_sagittal',
                    'largest_short_axis',
                ])
            elif 'softmax' in MODEL_NAME:
                logging.info(f"case_id: {case_id}, Node #: {node}, pos: {predict[0,0]}, ENE: {predict[0,1]}, Tumor Volume: {tumor_volume},Tumor Volume before dilation: {tumor_volume_before_dilation}, Largest Short Axis (mm): {largest_short_axis}")
                prediction_list.append([
                    case_id,
                    node,
                    predict[0, 0],
                    predict[0, 1],
                    tumor_volume,
                    tumor_volume_before_dilation,
                    short_axis_axial,
                    short_axis_coronal,
                    short_axis_sagittal,
                    long_axis_axial,
                    long_axis_coronal,
                    long_axis_sagittal,
                    largest_short_axis,
                ])
                df_pred = pd.DataFrame(prediction_list, columns=[
                    'case_id',
                    'node',
                    'pos',
                    'ene',
                    'tumor_volume_after_dilation',
                    "tumor_volume_before_dilation",
                    'short_axis_axial',
                    'short_axis_coronal',
                    'short_axis_sagittal',
                    'long_axis_axial',
                    'long_axis_coronal',
                    'long_axis_sagittal',
                    'largest_short_axis',
                ])

            # Optionally save resampled data only if short-axis computed
            if getattr(args, "include_short_axes", False):
                if not os.path.exists(os.path.join(output_folder, f"resampled_{timestamp}")):
                    os.makedirs(os.path.join(output_folder, f"resampled_{timestamp}"))
                sitk.WriteImage(isotropic_image_sitk, 
                               os.path.join(output_folder, f"resampled_{timestamp}", f"{case_id}_{node}_isotropic_image.nrrd"))
                sitk.WriteImage(isotropic_mask_sitk, 
                               os.path.join(output_folder, f"resampled_{timestamp}", f"{case_id}_{node}_isotropic_mask.nrrd"))

logging.info("Final Prediction List:")
logging.info(str(prediction_list))

# Ensure prediction output directory exists (CSV will be saved below)
if not os.path.exists(prediction_folder):
    os.makedirs(prediction_folder)

# Build DataFrame once after processing, guard empty
if len(prediction_list) > 0:
    # Column semantics:
    # - pos: model probability of the non-ENE (negative) class
    # - ene: model probability of the ENE (positive) class
    # - sigmoid head branch includes 'not_ene' where not_ene = 1 - ene
    if 'multi' in MODEL_NAME:
        all_columns = ['case_id','node','pos','ene','tumor_volume_after_dilation','tumor_volume_before_dilation',
                       'short_axis_axial','short_axis_coronal','short_axis_sagittal','long_axis_axial','long_axis_coronal','long_axis_sagittal','largest_short_axis']
    elif 'sigmoid' in MODEL_NAME and 'multi' not in MODEL_NAME:
        all_columns = ['case_id','node','not_ene','ene','tumor_volume_after_dilation','tumor_volume_before_dilation',
                       'short_axis_axial','short_axis_coronal','short_axis_sagittal','long_axis_axial','long_axis_coronal','long_axis_sagittal','largest_short_axis']
    else:  # softmax fallback
        all_columns = ['case_id','node','pos','ene','tumor_volume_after_dilation','tumor_volume_before_dilation',
                       'short_axis_axial','short_axis_coronal','short_axis_sagittal','long_axis_axial','long_axis_coronal','long_axis_sagittal','largest_short_axis']

    df_pred = pd.DataFrame(prediction_list, columns=all_columns)
    # Add ENE classification next to 'ene' with threshold 0.3
    if 'ene' in df_pred.columns:
        ene_idx = df_pred.columns.get_loc('ene')
        df_pred.insert(ene_idx + 1, 'ene_classification', np.where(df_pred['ene'] > 0.3, 1, 0))

    # # Add a note column to explain synthetic node id -1 rows
    # if 'node' in df_pred.columns:
    #     df_pred['note'] = ""
    #     df_pred.loc[df_pred['node'] == -1, 'note'] = (
    #         "node = -1: case has no malignant node detected (no label 2 foreground); "
    #         "all prediction and measurement values are set to 0."
    #     )

    if not getattr(args, "include_short_axes", False):
        axis_cols = ['short_axis_axial','short_axis_coronal','short_axis_sagittal',
                     'long_axis_axial','long_axis_coronal','long_axis_sagittal','largest_short_axis']
        existing = [c for c in axis_cols if c in df_pred.columns]
        if len(existing) > 0:
            df_pred = df_pred.drop(columns=existing)

    df_pred.to_csv(os.path.join(prediction_folder, f"prediction_{timestamp}.csv"), index=False)
else:
    logging.warning("No predictions to save; CSV not written.")

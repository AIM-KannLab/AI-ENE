import subprocess
import os

def rtstruct_to_nrrd(dataset, patient_id, path_to_rtstruct, path_to_image, output_dir, prefix = ""):
    """
    Converts a single rtstruct file into a folder containing individual structure
    nrrd files. The folder will be named dataset_patient id.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_rtstruct (str): Path to the rtstruct file.
        path_to_image (str): Path to the image (.nrrd) associated with this rtstruct file.This is needed to match the size and dimensions of the image.
        output_dir (str): Path to folder where the folder containing nrrds will be saved.
        prefix (str): If multiple rtstruct files belong to one patient, their contents can be saved in multiple folders using this prefix. If "", only one folder will be saved.

    Returns:
        None
    Raises:
        Exception if an error occurs.
    """
    if prefix == "":
        output_folder = os.path.join(output_dir, "{}_{}".format(dataset, patient_id))
    else:
        output_folder = os.path.join(output_dir, "{}_{}_{}".format(dataset, patient_id, prefix))
    cmd = ["plastimatch", "convert", "--input", path_to_rtstruct, "--output-prefix",
    output_folder, "--prefix-format", "nrrd", "--fixed", path_to_image]
    try:
        subprocess.call(cmd)
    except Exception as e:
        print ("dataset:{} patient_id:{} error:{}".format(dataset, patient_id, e))

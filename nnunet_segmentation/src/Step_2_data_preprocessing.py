import os
import yaml
from dicom_to_nrrd import run_dcm_to_nrrd
from image_preprocess import run_image_preprocessing
from prepare_nnUNet_data import get_nnUNet_data


def main():

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    def resolve_path(path_value, default_value=None):
        if not path_value:
            return default_value
        expanded = os.path.expanduser(path_value)
        if os.path.isabs(expanded):
            return expanded
        return os.path.abspath(os.path.join(repo_root, expanded))

    project_dir = resolve_path(config['path'].get('project_dir'), repo_root)
    # Flexible input handling
    input_data_dir = resolve_path(config['path'].get('input_data_dir'), None)
    dicom_dir_cfg = resolve_path(config['path'].get('dicom_dir'), os.path.join(repo_root, 'dicom_data'))
    input_format = (config.get('data', {}) or {}).get('input_data_format', 'dicom')
    crop_size_cfg = (config.get('data', {}) or {}).get('image_crop_size', [160, 160, 64])
    # Optional external registration template
    reg_template_cfg = resolve_path(config['path'].get('registration_template_path'), None)
    task_name = config['inference']['task_name']


    main_data_dir = os.path.join(project_dir, 'main_data')
    nrrd_img_dir = os.path.join(main_data_dir, 'raw_nrrd_image')
    respace_img_dir = os.path.join(main_data_dir, 'respace_image')
    crop_img_dir = os.path.join(main_data_dir, 'crop_image')
    # Default template under project if external not provided
    default_template = os.path.join(main_data_dir, 'reg_img_template', 'template_image.nrrd')
    reg_img_template_path = reg_template_cfg or default_template
    # Save nnU-Net data under the task directory; the preparation function will create 'imagesTs' inside
    nnUNet_task_dir = os.path.join(project_dir, 'nnUNet_output/nnUNet_raw_data', task_name)

    # Convert DICOM if requested; otherwise assume input_data_dir contains NRRD/NIfTI
    if input_format == 'dicom':
        run_dcm_to_nrrd(
            dicom_dir=dicom_dir_cfg,
            nrrd_dir=nrrd_img_dir)
        src_input_dir = nrrd_img_dir
    else:
        if not input_data_dir:
            raise ValueError("For non-DICOM inputs, 'path.input_data_dir' must be set in config.yaml")
        src_input_dir = input_data_dir

    run_image_preprocessing(
        input_image_format=input_format,
        input_data_dir=src_input_dir,
        raw_img_dir=nrrd_img_dir,
        respace_img_dir=respace_img_dir,
        crop_img_dir=crop_img_dir,
        reg_img_template_path=reg_img_template_path,
        crop_shape=tuple(crop_size_cfg))

    # Build nnU-Net data; do not write auxiliary CSV index by default
    get_nnUNet_data(data_dir=crop_img_dir, save_dir=nnUNet_task_dir, write_index=False)


if __name__ == '__main__':

    main()

    



    

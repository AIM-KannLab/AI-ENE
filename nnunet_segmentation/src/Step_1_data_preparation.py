import os
import yaml


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
    task_name = config['inference']['task_name']
    registration_template_path = resolve_path(config['path'].get('registration_template_path'), None)

    main_data_dir = os.path.join(project_dir, 'main_data')
    nnUNet_output_dir = os.path.join(project_dir, 'nnUNet_output')

    # No longer creating nnUNet_raw_data_base or nnUNet_preprocessed for inference-only pipeline

    # Store trained model under project root (not under nnUNet_output)
    nnUNet_trained_model_dir = os.path.join(project_dir, 'nnUNet_trained_model')
    os.makedirs(nnUNet_trained_model_dir, exist_ok=True)

    nnUNet_model_weight_dir = os.path.join(nnUNet_trained_model_dir, 'nnUNet', task_name, 'fold_0')
    os.makedirs(nnUNet_model_weight_dir, exist_ok=True)
    # Do not pre-create imagesTs/predsTs under nnUNet_raw_data_base; Step 2 and 3 manage under nnUNet_raw_data

    # Only create a workspace template folder when no explicit template path is configured.
    # When registration_template_path is provided (runner sets repo template), we avoid creating per-run folders.
    if registration_template_path is None:
        reg_img_template_dir = os.path.join(main_data_dir, 'reg_img_template')
        os.makedirs(reg_img_template_dir, exist_ok=True)

    nrrd_img_dir = os.path.join(main_data_dir, 'raw_nrrd_image')
    os.makedirs(nrrd_img_dir, exist_ok=True)

    respace_img_dir = os.path.join(main_data_dir, 'respace_image')
    os.makedirs(respace_img_dir, exist_ok=True)

    crop_img_dir = os.path.join(main_data_dir, 'crop_image')
    os.makedirs(crop_img_dir, exist_ok=True)

    print('data preparation done!')


if __name__ == '__main__':

    main()



    
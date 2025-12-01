import os
import yaml

# Deterministic seeding for nnU-Net inference
try:
    import random
    import numpy as np
    import torch
    SEED = int(os.environ.get("NNUNET_SEED", "42"))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    # If torch is not available in this environment, skip seeding silently
    pass


def main():

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Resolve project_dir: support ~ and relative paths from the project root (JCO/)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    project_dir_raw = config['path'].get('project_dir', '.')
    project_dir_expanded = os.path.expanduser(project_dir_raw)
    if os.path.isabs(project_dir_expanded):
        project_dir = project_dir_expanded
    else:
        project_dir = os.path.abspath(os.path.join(repo_root, project_dir_expanded))
    task_name = config['inference']['task_name']


    nnUNet_output_dir = os.path.join(project_dir, 'nnUNet_output')
    # Drop nnUNet_raw_data_base/nnUNet_preprocessed for inference-only pipeline
    # Read trained model from project root
    nnUNet_trained_model_dir = os.path.join(project_dir, 'nnUNet_trained_model')
    nnUNet_model_dir = os.path.join(nnUNet_trained_model_dir, 'nnUNet', task_name)
    # Read inputs directly from nnUNet_output/nnUNet_raw_data/<task_name>/imagesTs (no _raw_data_base)
    nnUNet_img_dir = os.path.join(nnUNet_output_dir, 'nnUNet_raw_data', task_name, 'imagesTs')
    nnUNet_pred_dir = os.path.join(nnUNet_output_dir, 'nnUNet_raw_data', task_name, 'predsTs')

    import sys
    nnUNet_dir = config['path'].get('nnUNet_dir', '')
    if nnUNet_dir:
        sys.path.append(os.path.expanduser(nnUNet_dir))

    # clear any inherited conflicting envs
    for k in ['RESULTS_FOLDER', 'nnUNet_results']:
        os.environ.pop(k, None)

    os.environ['RESULTS_FOLDER'] = nnUNet_trained_model_dir

    for var in ['RESULTS_FOLDER']:
        os.makedirs(os.environ[var], exist_ok=True)

    from nnunet.inference.predict import predict_from_folder


    predict_from_folder(
        model=nnUNet_model_dir,
        input_folder=nnUNet_img_dir,
        output_folder=nnUNet_pred_dir,
        folds=(0,),
        tta=False,
        overwrite_existing=True,
        save_npz=False,
        num_threads_preprocessing=6,
        num_threads_nifti_save=3,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
    )


if __name__ == '__main__':

    main()


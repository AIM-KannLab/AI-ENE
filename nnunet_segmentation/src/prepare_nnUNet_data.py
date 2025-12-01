import numpy as np
import os
import glob
import pandas as pd
import nibabel as nib
import SimpleITK as sitk


def get_nnUNet_data(data_dir, save_dir, write_index: bool = False):

    print('\n --- start preparing nnUNet data ---')

    img_ts_dir = save_dir + '/imagesTs'
    os.makedirs(img_ts_dir, exist_ok=True)

    img_crop_dirs = [i for i in sorted(glob.glob(data_dir + '/*nii.gz'))]
    img_dirs = []
    img_ids = []
    cohorts = []
    for img_dir in img_crop_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        cohort = img_id.split('_')[0]
        img_dirs.append(img_dir)
        img_ids.append(img_id)
        cohorts.append(cohort)
        print(img_id)
    df = pd.DataFrame({'cohort': cohorts, 'img_id': img_ids, 'img_dir': img_dirs})
    #print(df)
    
    ### test dataset: retain original IDs in filenames
    img_nn_ids = []
    for img_dir in df['img_dir']:
        # Preserve original base name (e.g., radcure_0005 -> radcure_0005_0000.nii.gz)
        img_id = os.path.basename(img_dir).split('.')[0]
        img_nn_id = f"{img_id}_0000.nii.gz"
        img_save_dir = img_ts_dir + '/' + img_nn_id
        img = sitk.ReadImage(img_dir)
        sitk.WriteImage(img, img_save_dir)
        img_nn_ids.append(img_nn_id)
    df['img_nn_id'] = img_nn_ids
    # Optional: write an index CSV only if requested
    if write_index:
        df.to_csv(os.path.join(save_dir, 'dataset_index.csv'), index=False)


if __name__ == '__main__':
    # Example usage (intentionally left empty to avoid hardcoded local paths)
    pass

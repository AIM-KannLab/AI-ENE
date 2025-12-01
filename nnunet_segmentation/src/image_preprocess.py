import sys
import os
import pydicom
import glob
import SimpleITK as sitk
import pandas as pd
import numpy as np
from interpolate import interpolate
from crop_image import crop_top, crop_top_image_only, crop_full_body
from registration import nrrd_reg_rigid
import SimpleITK as sitk
import shutil



def interpolation(input_image_format, input_data_dir, raw_img_dir, respace_img_dir):
    print('\n --- start image interpolation ---')

    # Source selection: if starting from DICOM, we expect converted NRRDs in raw_img_dir
    if input_image_format == 'dicom':
        img_glob_dir = raw_img_dir
        patterns = ['*.nrrd']
    else:
        img_glob_dir = input_data_dir
        if input_image_format in ['nrrd']:
            patterns = ['*.nrrd']
        elif input_image_format in ['nii', 'nii.gz']:
            # Support both compressed and uncompressed NIfTI
            patterns = ['*.nii', '*.nii.gz']
        else:
            raise ValueError(f"Unsupported input_image_format: {input_image_format}")

    img_dirs = []
    for pat in patterns:
        img_dirs.extend(sorted(glob.glob(os.path.join(img_glob_dir, pat))))

    count = 0
    for img_dir in img_dirs:
        img_id = os.path.basename(img_dir).split('.')[0]
        count += 1
        print(count, img_id)
        # Interpolate to 1x1x3 spacing; interpolate() routine handles reading NRRD/NIfTI via SimpleITK
        interpolate(
            patient_id=img_id,
            path_to_nrrd=img_dir,
            interpolation_type='linear',
            new_spacing=(1, 1, 3),
            return_type='sitk_obj',
            output_dir=respace_img_dir,
            img_format='nrrd')


def reg_crop(crop_shape, respace_img_dir, crop_img_dir, reg_img_template_path):

    print('\n --- start image registration and cropping ---')

    IDs = []
    for dir in sorted(glob.glob(crop_img_dir + '/*nii.gz')):
        ID = dir.split('/')[-1].split('.')[0]
        IDs.append(ID)

    img_dirs = [i for i in sorted(glob.glob(respace_img_dir + '/*nrrd'))]
    img_ids = []
    bad_ids = []
    count = 0
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        if img_id not in IDs:
            count += 1
            print(count, img_id)
            img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
            # --- crop full body scan ---
            z_img = img.GetSize()[2]
            if z_img > 200:
                img = crop_full_body(img, int(z_img * 0.65))
            # --- registration for image and seg ---    
            
            fixed_img = sitk.ReadImage(reg_img_template_path, sitk.sitkFloat32)
            try:
                # register images
                reg_img, fixed_img, moving_img, final_transform = nrrd_reg_rigid( 
                    patient_id=img_id, 
                    moving_img=img, 
                    output_dir='', 
                    fixed_img=fixed_img,
                    image_format='nrrd')
                # crop
                crop_top_image_only(
                    patient_id=img_id,
                    img=reg_img,
                    crop_shape=crop_shape,
                    return_type='sitk_object',
                    output_dir=crop_img_dir,
                    image_format='nii.gz')
            except Exception as e:
                bad_ids.append(img_id)
                print(img_id, e)
    #print('bad ids:', bad_ids)


def run_image_preprocessing(input_image_format, input_data_dir, raw_img_dir, respace_img_dir, crop_img_dir, reg_img_template_path, crop_shape):

    interpolation(input_image_format, input_data_dir, raw_img_dir, respace_img_dir)

    reg_crop(crop_shape, respace_img_dir, crop_img_dir, reg_img_template_path)





if __name__ == '__main__':
    # Example usage intentionally omitted to avoid hardcoded local paths
    pass

    





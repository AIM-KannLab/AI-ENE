import sys, os, glob
import SimpleITK as sitk
import pydicom
import numpy as np



def load_dicom(slice_list):

    #print(slice_list)
    img_dirs = []
    for img_dir in slice_list:
        img_type = img_dir.split('/')[-1].split('.')[0]
        if img_type not in ['RTDOSE', 'RTSTRUCT']:
            img_dirs.append(img_dir)
    slices = []
    for s, t in zip(img_dirs, img_dirs[1:]):
        slice1 = pydicom.read_file(s, force=True)
        slice2 = pydicom.read_file(t, force=True)
        #print(slice.ImageOrientationPatient)
        #print(slice.SliceThickness)
        if slice1.ImageOrientationPatient==[0, 1, 0, 0, 0, -1]:
            slice1.ImageOrientationPatient=[1, 0, 0, 0, 1, 0]
            slice2.ImageOrientationPatient=[1, 0, 0, 0, 1, 0]
        if float(slice1.SliceThickness) == np.abs(slice1.ImagePositionPatient[2] - slice2.ImagePositionPatient[2]):
            slices.append(slice1)
        elif slice1.ImageOrientationPatient==[1, 0, 0, 0, 1, 0]:    
            slices.append(slice1)
        if t == slice_list[-1]:
            slices.append(slice2)
    #try:
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        #try:
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    if slice_thickness > 3 or slice_thickness == 0:
        slice_thickness = np.abs(slices[9].ImagePositionPatient[2] - slices[10].ImagePositionPatient[2])
#        except:
#            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
#            if slice_thickness > 3:
#                slice_thickness = np.abs(slices[9].SliceLocation - slices[10].SliceLocation)
#    except Exception as e:
#        print (e)
#        print('No position found for image', slice_list[0])
#        if 'Sinai'  or 'TCGA' or 'E3311' in slice_list[0]:
#            slice_thickness = float(slices[0].SliceThickness)
#            print("fall back slice thickness: ", slice_thickness)
#        else:
#            #slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
#            #print("slice thickness: ", slice_thickness)
#            return []
    for s in slices:
        s.SliceThickness = slice_thickness
        #print('s.SliceThickness: ', s.SliceThickness, s.SpacingBetweenSlices)
        #print('image position:', s.ImagePositionPatient)
    img_spacing = [float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), slice_thickness]
    #print('img_spacing: ', img_spacing)
    img_direction = [float(i) for i in slices[0].ImageOrientationPatient] + [0, 0, 1]
    img_origin = slices[0].ImagePositionPatient
    
    return slices, img_spacing, img_direction, img_origin


def getPixelArray(slices):

    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def run_core(dicom_dir, image_format):

    #dicomFiles = sorted(glob.glob(dicom_dir + '/*.dcm'))
    dicomFiles = sorted(glob.glob(dicom_dir + '/[!D]*'))
    dicomCheck = list(map(lambda sub:int(''.join([i for i in sub if i.isnumeric()])), dicomFiles)) 
    #Extracts only the numbers to check for duplicates
    if len(dicomCheck) > len(set(dicomCheck)):
        dicomFiles = [item for item in dicomFiles if '.dcm' not in item]
        print("Removing duplicate slices")
    if image_format=='ct':
        slices, img_spacing, img_direction, img_origin = load_dicom(dicomFiles)
        print('img_spacing: ', img_spacing)

    if 0.0 in img_spacing:
        print ('ERROR - Zero spacing found for patient,', img_spacing)
        return ''
    if image_format=='ct':
        imgCube = getPixelArray(slices)

    # Build the SITK nrrd image
    imgSitk = sitk.GetImageFromArray(imgCube)
    imgSitk.SetSpacing(img_spacing)
    imgSitk.SetDirection(img_direction)
    imgSitk.SetOrigin(img_origin)
    
    return imgSitk


def dcm_to_nrrd(patient_id, dicom_dir, output_dir, image_format, save=True):
    """
    Converts a stack of slices into a single .nrrd file and saves it.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        data_type (str): Type of data (e.g., ct, pet, mri..)
        input_dir (str): Path to folder containing slices.
        output_dir (str): Path to folder where nrrd will be saved.
        save (bool): If True, the nrrd file is saved
    Returns:
        The sitk image object.
    Raises:
        Exception if an error occurs.
    """

    nrrd_file_path = output_dir + '/' + patient_id + '.nrrd'
    sitk_object = run_core(dicom_dir, image_format)
    if save:
        nrrdWriter = sitk.ImageFileWriter()
        nrrdWriter.SetFileName(nrrd_file_path)
        nrrdWriter.SetUseCompression(True)
        nrrdWriter.Execute(sitk_object)
    print('DCM files have been successfully converted to nrrd format!')
    return sitk_object


def run_dcm_to_nrrd(dicom_dir, nrrd_dir):

    print('\n --- start converting dicom to nrrd ---')

    count = 0
    for folder in sorted(os.listdir(dicom_dir)):
        if not folder.startswith('.'):
            case_id = str(folder)
            count += 1
            print('\n', count, case_id)

            dcm_dir = dicom_dir + '/' + folder

            save_file_path = nrrd_dir + '/' + case_id + '.nrrd'
            if os.path.exists(save_file_path):
                print('nrrd file existed, moving to next one!!!')
            else:
                try:
                    dcm_to_nrrd(
                        patient_id=case_id,
                        dicom_dir=dcm_dir,
                        output_dir=nrrd_dir,
                        image_format='ct',
                        save=True)
                except Exception as e:
                    print(case_id, e)


if __name__ == '__main__':
    # Example usage intentionally omitted to avoid hardcoded local paths
    pass




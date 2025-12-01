import SimpleITK as sitk
import numpy as np
import os

def resize_x(arr, new_x, padding_value):
    old_x = arr.shape[2]
    if old_x == new_x:
        return arr
    elif  old_x > new_x:
        start_x = (old_x-new_x)//2
        return arr[:,:,start_x:start_x+new_x]
    else:
        padding_side_1 = (new_x-old_x)//2
        padding_side_2 = padding_side_1 + new_x - (padding_side_1*2 + old_x)
        return np.pad(arr, ((0,0),(0,0),(padding_side_1, padding_side_2)), mode='constant', constant_values=(padding_value))

def resize_y(arr, new_y, padding_value):
    old_y = arr.shape[1]
    if old_y == new_y:
        return arr
    elif  old_y > new_y:
        start_y = (old_y-new_y)//2
        return arr[:,start_y:start_y+new_y,:]
    else:
        padding_side_1 = (new_y-old_y)//2
        padding_side_2 = padding_side_1 + new_y - (padding_side_1*2 + old_y)
        return np.pad(arr, ((0,0),(padding_side_1, padding_side_2),(0,0)), mode='constant', constant_values=(padding_value))

def resize_z(arr, new_z, padding_value):
    old_z = arr.shape[0]
    if old_z == new_z:
        return arr
    elif  old_z > new_z:
        start_z = (old_z-new_z)//2
        return arr[start_z:start_z+new_z,:,:]
    else:
        padding_side_1 = (new_z-old_z)//2
        padding_side_2 = padding_side_1 + new_z - (padding_side_1*2 + old_z)
        return np.pad(arr, ((padding_side_1, padding_side_2),(0,0),(0,0)), mode='constant', constant_values=(padding_value))


def resize(dataset, patient_id, data_type, path_to_nrrd, shape, padding_value, return_type, output_dir = ""):
    """
    Resizes a given nrrd file to a given size in all three dimensions.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        data_type (str): Type of data (e.g., ct, pet, mri, lung(mask), heart(mask)..)
        path_to_nrrd (str): Path to nrrd file.
        shape (str): Tuple containing 3 values for size to resize to: (x,y,z).
        padding_value (int): Value to insert when padding is neccesary. For example, use -1024 for CT and 0 for masks.
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_dir (str): Optional. If provided, nrrd file will be saved there. If not provided, file will not be saved.
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type').
    Raises:
        Exception if an error occurs.
    """
    try:
        # load img
        img = sitk.ReadImage(path_to_nrrd)
        data = sitk.GetArrayFromImage(img)
        # resize in all directions
        print('Input shape: {}'.format(data.shape))
        data = resize_x(data, shape[0], padding_value)
        print('Shape after x: {}'.format(data.shape))
        data = resize_y(data, shape[1], padding_value)
        print('Shape after y: {}'.format(data.shape))
        data = resize_z(data, shape[2], padding_value)
        print('Shape after z: {}'.format(data.shape))
        new_sitk_object = sitk.GetImageFromArray(data)
        new_sitk_object.SetSpacing(img.GetSpacing())
        new_sitk_object.SetOrigin(img.GetOrigin())
        assert new_sitk_object.GetSize() == shape, "oops.. The shape of the returned array does not match your requested shape."
        if output_dir != "":
            writer = sitk.ImageFileWriter()
            writer.SetFileName(os.path.join(output_dir, "{}_{}_{}_interpolated_resized_raw_xx.nrrd".format(dataset, patient_id, data_type)))
            writer.SetUseCompression(True)
            writer.Execute(new_sitk_object)
        if return_type == "sitk_object":
            return new_sitk_object
        elif return_type == "numpy_array":
            return data
    except Exception as e:
        print ("Error in {}_{}, {}".format(dataset, patient_id, e))

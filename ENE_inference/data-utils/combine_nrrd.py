import SimpleITK as sitk
import util
import numpy as np

def combine_nrrd(list_of_nrrds, path_to_new_nrrd):
    # list_of_nrrds should have at least 2 items
    # will supress all labels to binary
    label_obj = sitk.ReadImage(list_of_nrrds[0])
    label_arr = sitk.GetArrayFromImage(label_obj)
    output_arr = np.zeros_like(label_arr)
    print (label_arr.shape)
    print (output_arr.shape)
    print (len(output_arr[output_arr==1]))
    for nrrd in list_of_nrrds:
        output_arr += sitk.GetArrayFromImage(sitk.ReadImage(nrrd))
        print (len(output_arr[output_arr==1]))

    output_arr[output_arr>1] = 1

    output_image  = sitk.GetImageFromArray(output_arr)

    output_image.SetSpacing(label_obj.GetSpacing())
    output_image.SetOrigin(label_obj.GetOrigin())

    util.write_sitk(output_image, path_to_new_nrrd)

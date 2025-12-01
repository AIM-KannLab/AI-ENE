import SimpleITK as sitk
import util


def subtract_nrrd(nrrd_subtract_from, nrrd_subtract, path_to_new_nrrd):
    # subtract from
    nrrd_subtract_from_obj = sitk.ReadImage(nrrd_subtract_from)
    nrrd_subtract_from_arr = sitk.GetArrayFromImage(nrrd_subtract_from_obj)
    # to subtract
    nrrd_subtract_arr = sitk.GetArrayFromImage(sitk.ReadImage(nrrd_subtract))
    # subtract
    nrrd_subtract_from_arr[nrrd_subtract_arr==1]=0
    # save
    output_image  = sitk.GetImageFromArray(nrrd_subtract_from_arr)
    output_image.SetSpacing(nrrd_subtract_from_obj.GetSpacing())
    output_image.SetOrigin(nrrd_subtract_from_obj.GetOrigin())
    util.write_sitk(output_image, path_to_new_nrrd)

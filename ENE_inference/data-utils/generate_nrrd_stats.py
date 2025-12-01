import SimpleITK as sitk
import pandas as pd
import os

def generate_nrrd_stats(dataset, input_dir, save=False):

    """
    Generates a pandas dataframe and/or csv of size, spacing, and range statistics
    from a folder of nrrd files (usually images).
    Args:
        dataset (str): Name of dataset.
        input_dir (string): Path to folder containing nrrd files.
        save (boolean): Boolean to save resulting dataframe as a csv
    Returns:
        pandas dataframe
    """

    columns = ['dataset','patient_id',
               'size_X','size_Y','size_Z',
               'spacing_X','spacing_Y','spacing_Z',
               'stack_min','stack_max']

    df = pd.DataFrame(columns=columns)
    df.head()

    for i, nrrd in enumerate(os.listdir(input_dir)):
        try:
            patient_id = nrrd.split("_")[1]
            nrrd = os.path.join(input_dir, nrrd)
            image = sitk.ReadImage(nrrd)
            arr = sitk.GetArrayFromImage(image)
            #
            size = image.GetSize()
            spacing = image.GetSpacing()
            _range = [arr.min(), arr.max()]
            #
            df.loc[i] = [dataset,patient_id,
                          size[0], size[1], size[2],
                          spacing[0], spacing[1], spacing[2],
                          _range[0], _range[1]]

            print ("{}_{} read".format(dataset, patient_id))
        except Exception as e:
            df.loc[i] = [dataset,patient_id] + ['' for x in range(8)]
            print ("Error in {}_{}, {}".format(dataset, patient_id, e))
    if save:
        df.to_csv(dataset + '_stats.csv')
    return df

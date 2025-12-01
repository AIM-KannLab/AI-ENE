import SimpleITK as sitk
import pandas as pd
import os
import util

def generate_structure_stats(dataset, patient_id_list, path_to_masks_list, save=False):

    """
    Calculates bounding box of all masks given and generates a pandas dataframe and/or csv of margins of either side of this box, as well as its dimensions.
    Args:
        dataset (str): Name of dataset.
        patient_id (list): list of Unique patient ids.
        path_to_masks_list (list): List of lists containing paths to mask nrrd files. Should match the patient_id list.
        save (boolean): Boolean to save resulting dataframe as a csv
    Returns:
        pandas dataframe
    """

    columns = ['dataset','patient_id',
               'right_margin_2_start','left_margin_2_end',
               'right_left_2',
               'anterior_margin_1_start','posterior_margin_1_end',
               'anterior_posterior_1',
               'inferior_margin_0_start', 'superior_margin_0_end', 'inferior_superior_0']


    # [0, 1, 2]
    # [inferior:superior, anterior:posterior, right:left]

    df = pd.DataFrame(columns=columns)
    df.head()

    for i, (patient_id, mask_list) in enumerate(zip(patient_id_list, path_to_masks_list)):
        print (i, patient_id, mask_list)
        try:
            mask_arr_list = [sitk.GetArrayFromImage(sitk.ReadImage(x)) for x in mask_list]
            combined = util.combine_masks(mask_arr_list)
            bbox = util.get_bbox(combined)
            #
            df.loc[i] = [dataset,patient_id,
                        bbox[4], combined.shape[2]-bbox[5], bbox[8],
                        bbox[2], combined.shape[1]-bbox[3], bbox[7],
                        bbox[0], combined.shape[0]-bbox[1], bbox[6]
                        ]

            print ("{}_{} read".format(dataset, patient_id))
        except Exception as e:
            df.loc[i] = [dataset,patient_id] + ['' for x in range(9)]
            print ("Error in {}_{}, {}".format(dataset, patient_id, e))
    if save:
        df.to_csv(dataset + '_stats.csv')
    return df

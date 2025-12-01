import pandas as pd
import os
import shutil

def generate_structure_names(dataset, input_dir):
    """
    Generates a .csv file with patients (rows) and structures (columns). The
    table is essentially a binary matrix displaying which structures are
    available for each patient.
    Args:
        dataset (str): Name of dataset.
        input_dir (str): Path to folder of folders:
        input_dir
        ├── patient_1
        |   ├── BODY.nrrd
        |   ├── brain.nrrd
        |   └── CTV.nrrd
        └── patient_2
            ├── Body.nrrd
            ├── BRAIN.nrrd
            └── GTV.nrrd
    Returns:
        None
    """
    # go through the structure and make sure there are no folders in the lowest level.
    # We need to do this first separetly.
    for folder in os.listdir(input_dir):
        path_1 = os.path.join(input_dir, folder)
        for structure in os.listdir(path_1):
            path_2 = os.path.join(path_1, structure)
            # check if folder
            if os.path.isdir(path_2):
                # loop through contents
                for substructure in os.listdir(path_2):
                    _from = os.path.join(path_2, substructure)
                    _to = os.path.join(path_1, "{}_{}".format(structure,substructure))
                    shutil.move(_from, _to)
                # delete that folder
                shutil.rmtree(path_2)


    # generate dictionary
    # key: dataset_patientID
    # value: list of all structures
    d = {}
    all_structures = []
    for folder in os.listdir(input_dir):
        path_1 = os.path.join(input_dir, folder)
        _key = folder
        _value = [structure for structure in os.listdir(path_1)]
        d[_key] = _value
        all_structures.extend(_value)

    # remove duplicates
    all_structures_no_duplicates = list(set(all_structures))

    # generate pandas dataframe and columns
    columns = ["dataset_patient"] + all_structures_no_duplicates
    df = pd.DataFrame(columns=columns)

    # generate binary matrix
    for i, patient in enumerate(d):
        binary = [1 if column in d[patient] else 0 for column in columns[1:]]
        df.loc[i] = [patient] +  binary

    # save
    df.to_csv("{}_structures.csv".format(dataset))

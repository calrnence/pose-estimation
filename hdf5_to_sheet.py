'''
Sample Usage:-
python hdf5_to_sheet.py --dir transformations_2024-08-01_11-40-41.hdf5
'''
import h5py 
import pandas as pd
import argparse
from pathlib import Path

def export(filename):
    with h5py.File(filename, 'r') as file:
        excel = filename.stem + '.xlsx'
        with pd.ExcelWriter(excel, engine='openpyxl') as writer:
            for group in file.keys():
                marker = file[group]
                dfs = {}
                # retrieve all data 
                for dataset in marker.keys():
                    dfs[dataset] = pd.DataFrame(marker[dataset][:,1], columns = [dataset], index = marker[dataset][:,0])
                # merge dataframes with the timestamps as a common index
                merged_df = pd.concat(dfs.values(), axis=1)
                merged_df.index.name = 'time elapsed'
                merged_df.to_excel(writer, sheet_name=group)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Path to HDF5 for reading")
    args = vars(ap.parse_args())

    filename= Path(args["dir"])
    

    export(filename)
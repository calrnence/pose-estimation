'''
Sample Usage:-
python visualization.py --dir transformations_2024-07-31_17-59-31.hdf5 --mrk 1
'''
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Path to HDF5 for reading")
    ap.add_argument("-m", "--mrk", required=True, help="ID of marker to be visualized (integer)")
    args = vars(ap.parse_args())

    hdf5_path = args["dir"]
    marker_id = args["mrk"]

    with h5py.File(hdf5_path, 'r') as file:
        group = file[f"marker_{marker_id}"]
        # check if marker was detected
        if f"marker_{marker_id}" not in file.keys():
            print(f"Marker {marker_id} was not found.")
            sys.exit(0)
        
        figure, axes = plt.subplots(2, 3, num=f'marker {marker_id} data')
        axes = axes.flatten()

        for i, key in enumerate(group.keys()):
            data = group[key][:]
            axes[i].plot(data[:,0], data[:,1])
            axes[i].set_title(f"{key} data")
            axes[i].set_xlabel("time elapsed")
            axes[i].set_ylabel(f"{key}")
        plt.show()   

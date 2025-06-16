import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
from numpy.typing import NDArray 
import h5py
import sys

def read_data(file_path: str
                   ) -> tuple[NDArray[float], NDArray[int]]:
    """
    Reads fluxes and labels from HDF5 file.
    HDF5 file must contain following dataset: fluxes, labels.

    Parameters:
        file_path (str): path to HDF5 file.
    
    Returns:
        Tuple[NDArray[float], NDArray[int]]:
            2D array with preprocessed fluxes.
            1D array of labels.
    """
    with h5py.File(file_path, "r") as h5f:
        fluxes = h5f["fluxes"][:]
        labels = h5f["labels"][:]
        
    return fluxes, labels

def main(config_path: str, result_dir_path: str) -> None:
    """
    Loads config, reads data from provided HDF5 file, applies t-SNE on data, then saves result.
    
    Parameters:
        config_path (str): path to config file.
        result_dir_path (str): path to directory, where result will be saved.
    """
    with open(config_path) as f:
        config = json.load(f)
    
    fluxes, labels = read_data(config["data_path"])
    classes = config["classes"]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    fluxes_embedded = tsne.fit_transform(fluxes)

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(classes):
        mask = (labels == i)
        plt.scatter(fluxes_embedded[mask, 0], fluxes_embedded[mask, 1], label=class_name)

    plt.title("t-SNE vizualization")
    plt.savefig(result_dir_path + "/t-sne.png")


    data = {
        "x": fluxes_embedded[:,0].tolist(),
        "y": fluxes_embedded[:,1].tolist(),
        "labels": labels.tolist(),
        "classes": classes
    }

    with open(result_dir_path + "/dim_reduc.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Enter only path to config and path to save results")
    main(sys.argv[1], sys.argv[2])
    

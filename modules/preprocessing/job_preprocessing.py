import json
import sys
import numpy as np
from astropy.io import fits
from sklearn.preprocessing import minmax_scale
from pathlib import Path
import h5py
from numpy.typing import NDArray 

def read_spectrum(file_path: str) -> tuple[str, NDArray[float], NDArray[float]]:
    """
    Reads LAMOST DR2 spectrum from FITS file.

    Parameters:
        file_path (str): path to LAMOST DR2 FITS file.
    
    Returns:
        Tuple[str, NDArray[float], NDArray[float]:
            Spectrum filename.
            1D array of raw spectrum wave.
            1D array of raw spectrum flux.
    """
    with fits.open(file_path) as hdul:
        filename = hdul[0].header["FILENAME"]
        wave = hdul[0].data[2]
        flux = hdul[0].data[0]

    return filename, wave, flux

def preprocess_lamost_dr2_dir(src_path: str, start: float, end: float, 
                              points: int) -> tuple[NDArray[str], NDArray[float], NDArray[NDArray[float]]]:
    """
    Preprocess directory containing LAMOST DR2 FITS files.

    Parameters:
        src_path (str): path to directory containing LAMOST DR2 FITS file.
        start (float): starting wavelength in angstroms.
        end (float): ending wavelength in angstroms.
        points (int): count of fluxes within the selected wavelength interval.

    Returns:
        Tuple[NDArray[str], NDArray[float], NDArray[NDArray[float]]]:
            1D array of spectra filenames.
            1D array of preprocessed spectra wave.
            2D array of preprocessed fluxes.
    """
    src_dir = Path(src_path)

    filename_list = []
    flux_list = []
    new_wave = np.linspace(start, end, points, dtype=float)
    for f in src_dir.iterdir():
        filename, wave, flux = read_spectrum(f)
        new_flux = np.interp(new_wave, wave, flux)
        filename_list.append(filename)
        flux_list.append(new_flux)
        
    filenames = np.array(filename_list)
    wave = np.array(new_wave, dtype=np.float64)
    fluxes = np.array(flux_list, dtype=np.float64)
    fluxes = minmax_scale(fluxes, feature_range=(-1, 1), axis=1, copy=False)

    return filenames, wave, fluxes

def write_preprocessed_data(file_path: str, filenames: NDArray[str], wave: NDArray[float], 
                            fluxes: NDArray[NDArray[float]]) -> None:
    """
    Writes the preprocessed data to HDF5 file.

    Parameters:
        file_path (str): path to HDF5, where training data will be written.
        filenames (NDArray[str]): 1D array containing spectrum filenames.
        wave (NDArray[float]): 1D array containing spectrum preprocessed wave.
        fluxes (NDArray[NDArray[float]]): 2D array containing spectrum prepricessed fluxes.
    """

    with h5py.File(file_path, "w") as h5f:
        h5f.create_dataset("filenames", data=filenames.tolist(), dtype=h5py.string_dtype("utf-8"))
        h5f.create_dataset("wave", data=wave)
        h5f.create_dataset("fluxes", data=fluxes)

def main(config_path: str, result_dir_path: str) -> None:
    """
    Loads config, then starts preprocessing.
    
    Parameters:
        config_path (str): path to config file.
        result_dir_path (str): path to directory, where result will be saved.
    """
    with open(config_path) as f:
        config = json.load(f)
    
    filenames, wave, fluxes = preprocess_lamost_dr2_dir(config["data_dir_path"], config["wave_start_point"], 
                                                   config["wave_end_point"], config["wave_point_count"])
    
    write_preprocessed_data(f'{result_dir_path}/result.h5', filenames, wave, fluxes)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Missing parameters")
    main(sys.argv[1], sys.argv[2])

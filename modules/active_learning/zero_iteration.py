import numpy as np
import json

from config import ActiveLearningConfig
import file_utils

def create_new_config(config: ActiveLearningConfig) -> None:
    """
    Creates and saves configuration for first iteration.

    Parameters:
        config (ActiveLearningConfig): job's configuration, loaded from configuration file.
    """

    new_config = config.model_dump(exclude={"result_dir_path"})
    new_config["iteration"] = config.iteration + 1
    new_config["pool_data_path"] = config.result_dir_path + "/result.h5"
    new_config["training_data_to_add_path"] = config.result_dir_path + "/result.h5"
    new_config["oracle_data_to_add_path"] = config.result_dir_path + "/oracle_data.json"

    with open(f"{config.result_dir_path}/new_config.json", 'w', encoding='utf-8') as f:
        json.dump(new_config, f, indent=4) 

def run(config: ActiveLearningConfig) -> None:
    """
    Runs zero iteration of active learning job.
    
    1. Loads pool data.
    2. Gets oracle indexes.
    3. Saves results to file and creates severel files.

    Parameters:
        config (ActiveLearningConfig): job's configuration, loaded from configuration file.
    """
    
    filenames, wave, fluxes = file_utils.read_pool_data(config.pool_data_path)
    oracle_indexes = np.arange(config.oracle_batch_size)

    spectra_fluxes = {}
    for i in oracle_indexes:
        spectra_fluxes[filenames[i]] = fluxes[i].tolist()
    prep_spectra = {
        "wave": wave.tolist(),
        "spectra": spectra_fluxes
    }

    result = {
        "filenames": filenames,
        "wave": wave,
        "fluxes": fluxes,
        "oracle_indexes": oracle_indexes
    }

    file_utils.write_active_learning_0_iter(config.result_dir_path+"/result.h5", result)
    with open(f"{config.result_dir_path}/prep_spectra.json", 'w', encoding='utf-8') as f:
        json.dump(prep_spectra, f)

    create_new_config(config)
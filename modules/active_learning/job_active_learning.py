import sys
import json

import zero_iteration
import regular_iteration
from config import ActiveLearningConfig

def main(config_path: str, result_dir_path: str) -> None:
    """
    Loads and validates config, then runs the iteration.
    
    Parameters:
        config_path (str): path to config file.
        result_dir_path (str): path to directory, where result will be saved.
    """
    with open(config_path) as f:
        config_file = json.load(f)

    config = ActiveLearningConfig.model_validate(config_file)
    config.result_dir_path = result_dir_path

    if config.iteration == 0:
        zero_iteration.run(config)
    else: 
        regular_iteration.run(config)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Enter only path to config and path to save results")
    main(sys.argv[1], sys.argv[2])
    

    

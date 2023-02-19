import subprocess
import os
import sys

from packages.utils.constants import LANGUAGES, SIGM_DATA_PATH, HALL_EXPANDED_DATA_PATH

# TODO: adjust this file
def generate_augmentations():
    if not os.path.exists(HALL_EXPANDED_DATA_PATH):
        os.mkdir(HALL_EXPANDED_DATA_PATH)
        
    for fname in os.listdir(SIGM_DATA_PATH):
        if fname[-3:] == "low" and fname.split('-')[0] in LANGUAGES:

            result = subprocess.run(["python", "augment.py",   
                SIGM_DATA_PATH, fname, # required arguments
                "--examples", "100000"], check=True) # optional arguments (default values
            if result.returncode != 0:
                print(f"Generating augmentations for fname {fname} failed")
                sys.exit(1)

generate_augmentations()
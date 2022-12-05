from packages.utils.constants import SIGM_DATA_PATH, HALL_DATA_PATH
import pandas as pd

def get_initial_generation_frame(language): # concatenation of test file and augmentation file.
    test_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-test", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    assert len(test_frame) in [100, 1000] # NOTE: if this fails, it will be problematic. Other code (e.g., RandomSampler assumes 100 datapoints trainng points)
    augmentation_frame = pd.read_csv(f"{HALL_DATA_PATH}/{language}-train-low-hall", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    assert len(augmentation_frame) == 10000 # NOTE: if this fails, it will be problematic. Other code (e.g., RandomSampler assumes 10,000 real datapoints)

    test_frame = pd.concat([test_frame, augmentation_frame]) # this is for the initial model, so we can get likelihoods and generations
    return test_frame.reset_index(drop=True)
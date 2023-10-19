from dotenv import dotenv_values

config = dotenv_values(".env")
root_path = config['ROOT_PATH']
scratch_path = config['SCRATCH_PATH']

DEVICE = 'cpu'
BATCH_SIZE = 10
SCRATCH_PATH = f"{scratch_path}/augmentation_subset_select"
SIGM_DATA_PATH = f"{root_path}/conll2018/task1/all"
FAIRSEQ_SCRIPTS_PATH = f"{root_path}/mixture-of-augmentations/packages/fairseq_utils"
HALL_DATA_PATH = "/home/fsamir8/projects/rrg-msilfver/fsamir8/mixture-of-augmentations/data/spreadsheets/augmentations" 
HALL_EXPANDED_DATA_PATH = "/home/fsamir8/projects/rrg-msilfver/fsamir8/mixture-of-augmentations/data/spreadsheets/augmentations_expanded" 
LANGUAGES=("bengali", "turkish", "finnish", "georgian", "arabic", "navajo", "spanish")
ALGORITHMS = ('random', 'uncertainty_sample', 'initial', 'diversity_sample')
INITIAL_MODEL_PARAMS = ('train_medium', 'rand_seed', 'aug_pool_size')
ANALYSIS_SCRATCH_PATH = "/home/fsamir8/scratch/augmentation_subset_select/analysis"
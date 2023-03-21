DEVICE = 'cpu'
BATCH_SIZE = 10
SCRATCH_PATH = "/home/fsamir8/scratch/augmentation_subset_select"
SIGM_DATA_PATH = "/home/fsamir8/projects/rrg-msilfver/fsamir8/conll2018/task1/all"
FAIRSEQ_SCRIPTS_PATH = "/home/fsamir8/projects/rrg-msilfver/fsamir8/mixture-of-augmentations/packages/fairseq_utils"
HALL_DATA_PATH = "/home/fsamir8/projects/rrg-msilfver/fsamir8/mixture-of-augmentations/data/spreadsheets/augmentations" 
HALL_EXPANDED_DATA_PATH = "/home/fsamir8/projects/rrg-msilfver/fsamir8/mixture-of-augmentations/data/spreadsheets/augmentations_expanded" 
LANGUAGES=("bengali", "turkish", "finnish", "georgian", "arabic", "navajo", "spanish")
ALGORITHMS = ('random', 'uncertainty_sample', 'initial', 'diversity_sample')
INITIAL_MODEL_PARAMS=('train_medium', 'rand_seed', 'aug_pool_size')
ANALYSIS_SCRATCH_PATH = "/home/fsamir8/scratch/augmentation_subset_select/analysis"


ST_2023 = "/project/rrg-msilfver/fsamir8/2023InflectionST/part1/data"
import re

R_PAT = r"r=\d\.\d+"
USE_SOFTMAX_PAT = r"use_softmax_normalizer=\w+$"
NUM_AUG_PAT = r"num_aug=\d+"
TRAIN_MEDIUM_PAT=r"train_medium=(True|False)"
K_MEANS_PAT=r"k=\d+"
ALL_PATS = [R_PAT, USE_SOFTMAX_PAT, NUM_AUG_PAT, TRAIN_MEDIUM_PAT, K_MEANS_PAT]
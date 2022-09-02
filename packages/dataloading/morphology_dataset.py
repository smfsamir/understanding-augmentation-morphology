from typing import Tuple, Optional, List
import pandas as pd
from torch.utils.data import Dataset

class SigmorphonDataset(Dataset):

    def __init__(self, sigm_df: pd.DataFrame):
        self.sigm_df = sigm_df
    
    def __len__(self):
        return self.sigm_df.shape[0]

    def __getitem__(self, index: int) -> Tuple[str, str, str]:
        row = self.sigm_df.iloc[index]
        return (row.src, row.tgt, row.tag, row.is_hallucinated)
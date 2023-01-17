import pdb
import pandas as pd

def add_copies(train_frame: pd.DataFrame):
    """Adds copy triples into the training data.

    Args:
        train_frame (pd.DataFrame): Gold standard training data.

    Returns:
        _type_: _description_
    """
    copy_srcs = pd.concat([train_frame['src'], train_frame['tgt']])
    copy_tags = pd.concat([pd.Series(['COPY'] * len(train_frame)), train_frame['tag']])
    copy_frame = pd.DataFrame({
        "src": copy_srcs,
        "tag": copy_tags,
        "tgt": copy_srcs 
    })
    return pd.concat([train_frame, copy_frame])

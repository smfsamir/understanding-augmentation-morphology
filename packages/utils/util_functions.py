from .constants import SCRATCH_PATH
import datetime

def map_list(func, arr):
    return list(map(func, arr))

def get_artifacts_path(filename):
    date = datetime.today().strftime('%Y-%m-%d')
    dest = "data/artifacts/{}/{}".format( date,filename)
    return dest 

def get_today_date_formatted():
    date = datetime.today().strftime('%Y-%m-%d')
    return date

def get_mask(tensors, mask_token):
    """Return a mask that is the same shape as tensors. True if entry at tensors matches the mask token.

    Args:
        tensors (torch.Tensor): _description_
        mask_token (int): _description_
    """
    return tensors == mask_token

def get_model_augment_path(language, augmentation_type, **kwargs):
    """

    Args:
        language (str): Language the model is being trained for.
        augmentation_type (str): Augmentation type for the model. 
    """
    kwarg_strs= [f"{k}={v}" for k, v in kwargs.items()]
    kwarg_strs.sort()
    kwarg_str = "_".join(kwarg_strs)
    path = f"{SCRATCH_PATH}/{language}/{augmentation_type}_{kwarg_str}"
    return path 
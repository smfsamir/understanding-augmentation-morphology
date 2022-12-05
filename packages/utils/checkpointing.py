from packages.utils.constants import SCRATCH_PATH

import torch
from torch.utils.checkpoint import checkpoint
import os

def checkpoint_model(model, fname="model.pt", path=f"{SCRATCH_PATH}/checkpoints"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        {
            "model_state_dict": model.state_dict()
        },
        f"{path}/{fname}" 
    )

def load_model(model, fname="model.pt", path=f"{SCRATCH_PATH}/checkpoints"):
    with open(f"{path}/{fname}", 'rb'):
        checkpoint = torch.load(
            f"{path}/{fname}"
        )
        return model.load_state_dict(checkpoint['model_state_dict']) 
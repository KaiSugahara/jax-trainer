import jax
from flax.training import checkpoints
from flax.core.frozen_dict import freeze

def save_flax_params(params, ckpt_dir="params", prefix="name_"):
    
    checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, prefix=prefix, target=params, step=0, overwrite=True)

def load_flax_params(ckpt_dir="params", prefix="name_"):

    params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, prefix=prefix, target=None)
    params = freeze(jax.device_put(params))
    return params
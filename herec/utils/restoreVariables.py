from flax.training import checkpoints
from glob import glob

def restoreVariables(run_id, epoch_id = None):

    paths = glob(f"checkpoint/{run_id}/variables")
    if len(paths) == 0: raise Exception(f"run_id {run_id} does not exist")

    return checkpoints.restore_checkpoint(ckpt_dir=f'checkpoint/{run_id}/variables/checkpoint_0', target=None)
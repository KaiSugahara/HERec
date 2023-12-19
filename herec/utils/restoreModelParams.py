from flax.training import checkpoints
from glob import glob
from herec.utils import *

def restoreModelParams(run_id, epoch_id = None):

    paths = glob(f"{getRepositoryPath()}/checkpoint/{run_id}/*[0-9]*")
    if len(paths) == 0: raise Exception(f"run_id {run_id} does not exist")
    
    saved_epochs = sorted(map(lambda path: int(path.rsplit("/", 1)[1]), paths))
    print(saved_epochs)

    if epoch_id is None:
        return checkpoints.restore_checkpoint(ckpt_dir=f'{getRepositoryPath()}/checkpoint/{run_id}/{min(saved_epochs)}/default/checkpoint', target=None)
    elif epoch_id == -1:
        return checkpoints.restore_checkpoint(ckpt_dir=f'{getRepositoryPath()}/checkpoint/{run_id}/{max(saved_epochs)}/default/checkpoint', target=None)
    else:
        if epoch_id not in saved_epochs: raise Exception(f"epoch {epoch_id} is not saved")
        return checkpoints.restore_checkpoint(ckpt_dir=f'{getRepositoryPath()}/checkpoint/{run_id}/{epoch_id}/default/checkpoint', target=None)
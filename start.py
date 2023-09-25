import argparse

from herec.utils import *
from herec.train import train

"""
    0. Parse Input Arguments
"""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '-m', "--model",
    choices=["MF", "HE_MF", "FM", "HE_FM", "MF_BPR", "HE_MF_BPR"],
    help='name of the model to be trained and tested',
    required=True,
)
parser.add_argument(
    '-d', "--dataset",
    choices=["ML100K", "ML1M", "ML10M", "ML25M", "Ciao", "Ciao_PART", "Twitch100K"],
    help='a dataset to be trained and tested',
    required=True,
    nargs="+",
)
parser.add_argument(
    "--config",
    help='path of config file (yaml file)',
    required=True,
)
parser.add_argument(
    "--seed",
    help='seed',
    required=True,
    type=int,
    nargs="+",
)
parser.add_argument(
    "--memo",
    help='memo stored in MLflow run',
)
args = parser.parse_args()

"""
    1. Training
"""

# Set Hyper-parameter Suggester
suggester = hyParamSuggester(args.config)

# Train on Specified datasets and seeds
for dataset_name in args.dataset:
    for seed in args.seed:

        train(
            model_name = args.model,
            dataset_name = dataset_name,
            suggester = suggester,
            seed = seed,
            memo = args.memo,
        )
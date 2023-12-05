import argparse

from herec.utils import *
from herec.train import train
from herec.test import test

"""
    0. Parse Input Arguments
"""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '-m', "--model",
    choices=[
        "MF", "HE_MF", "HSE_MF",
        "FM", "HE_FM", "HSE_FM",
        "MF_BPR", "HE_MF_BPR", "HE_MF_USER_BPR", "HE_MF_ITEM_BPR", "HSE_MF_BPR",
        "MF_BCE", "HE_MF_BCE", "HSE_MF_BCE",
        "MF_SSM", "HE_MF_SSM", "HSE_MF_SSM",
        "GRU4Rec", "HE_GRU4Rec", "HSE_GRU4Rec",
    ],
    help='name of the model to be trained and tested',
    required=True,
)
parser.add_argument(
    '-d', "--dataset",
    choices=[
        "ML100K", "ML1M", "ML10M", "ML25M", "Ciao", "Ciao_PART",
        "ML100K_IMPLICIT", "ML1M_IMPLICIT", "Twitch100K", "DIGINETICA",
        "AMAZON_M2", "G1NEWS_SESSION",
    ],
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

# Train/Test on Specified datasets and seeds
for dataset_name in args.dataset:
    for seed in args.seed:

        train(
            modelName = args.model,
            datasetName = dataset_name,
            suggester = suggester,
            seed = seed,
            memo = args.memo,
        )
        
        test(
            modelName = args.model,
            datasetName = dataset_name,
            seed = seed,
            memo = args.memo,
        )
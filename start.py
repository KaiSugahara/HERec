import argparse
from dotenv import load_dotenv
import mlflow

from herec.reader import *
from herec.utils import *
from herec.train_test import train_test

"""
    0. Parse Input Arguments
"""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '-m', "--model",
    choices=["MF", "HE_MF", "FM", "HE_FM"],
    help='name of the model to be trained and tested',
    required=True,
)
parser.add_argument(
    '-d', "--dataset",
    choices=["ML100K", "ML1M", "ML10M", "ML25M"],
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
args = parser.parse_args()

"""
    1. Set Hyper-parameter Suggester
"""

suggester = hyParamSuggester(args.config)

for DATASET in args.dataset:
    for SEED in args.seed:

        """
            2. Set a Reader of Specified Dataset
        """

        if DATASET == "ML100K":
            reader = ML100K()
        elif DATASET == "ML1M":
            reader = ML1M()
        elif DATASET == "ML10M":
            reader = ML10M()
        elif DATASET == "ML25M":
            reader = ML25M()

        """
            3. Setup MLFlow
        """

        load_dotenv(".env")

        EXPERIMENT_NAME = f"HeRec-{args.model}-{DATASET}"
        if (experiment := mlflow.get_experiment_by_name(EXPERIMENT_NAME)) is None:
            experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
        else:
            experiment_id = experiment.experiment_id

        print("実験名:", EXPERIMENT_NAME)
        print("実験ID:", experiment_id)

        """
            4. Training and Testing
        """

        train_test(args.model, reader, suggester, SEED, experiment_id)
"""
    Parse Arguments
"""

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '-m', "--model",
    choices=[
        "IHSR",
        "eTREE",
    ],
    help='name of the model to be trained and tested',
    required=True,
)
parser.add_argument(
    '-d', "--dataset",
    choices=[
        # Explicit RS
        "ML100K", "ML1M", "Ciao", "Ciao_PART", "Yelp",
    ],
    help='a dataset to be trained and tested',
    required=True,
    nargs="+",
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
parser.add_argument(
    "--threads",
    help='threads',
    required=True,
    type=int,
    default=30,
)
args = parser.parse_args()

"""
    Test
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
os.environ["MKL_NUM_THREADS"] = str(args.threads)
os.environ["VECLIB_NUM_THREADS"] = str(args.threads)

import sys
import optuna
import numpy as np
from scipy.sparse import csr_array
import math

sys.path.append("..")
from herec.utils import *

from eTREE import eTREE
from IHSR import IHSR

from dotenv import load_dotenv
import mlflow
load_dotenv("../.env")

class test:
    
    def evaluate(self):

        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            
            hyparams = self.hyparams.copy()

            # Save Hyper-parameter to MLFlow
            mlflow.log_params(hyparams["model"])
            mlflow.log_dict(hyparams, "params.json")

            # Save memo to MLFlow
            if self.memo is not None:
                mlflow.set_tag('memo', self.memo)
        
            # Define Model
            if self.modelName == "eTREE":
                model = eTREE(
                    R = hyparams["model"]["R"],
                    item_clusters = {"A": [10], "B": [25, 5], "C": [50, 10, 3]}[hyparams["model"]["item_clusters"]],
                    lbd = hyparams["model"]["lbd"],
                    mu = hyparams["model"]["mu"],
                    eta = hyparams["model"]["eta"],
                    seed = self.seed,
                    maxNumIter = self.epochNum,
                    es = False,
                    run = run,
                )
            elif self.modelName == "IHSR":
                model = IHSR(
                    d = hyparams["model"]["d"],
                    n_by_level = {1: hyparams["model"]["userClusterNum"]},
                    m_by_level = {1: hyparams["model"]["itemClusterNum"]},
                    lam = hyparams["model"]["lam"],
                    seed = self.seed,
                    max_iter = self.epochNum,
                    es = False,
                    run = run,
                )

            # Train
            model.fit(self.X, self.W, self.X_VALID, self.W_VALID)
            print()

        return self

    def readyMLflow(self):

        load_dotenv(".env")

        EXPERIMENT_NAME = f"{self.datasetName}-{self.modelName}-TEST"
        if (experiment := mlflow.get_experiment_by_name(EXPERIMENT_NAME)) is None:
            self.experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
        else:
            self.experiment_id = experiment.experiment_id

        print("Experiment Name:", EXPERIMENT_NAME)
        print("Experiment ID:", self.experiment_id)

        return self
    
    def dowmload_best_setting(self):

        # Get Validation Results
        result_df = resultLoader( f"{self.datasetName}-{self.modelName}-TRAIN" ).get_results_by_fold( self.seed )
        
        # Check
        if result_df.height != 100:
            raise Exception("Please evaluate test set after valiadation")

        # Get Hyparam. Setting with Best Validation Score
        setting_path = result_df.get_column("artifact_uri")[0] + "/params.json"
        hyparams = mlflow.artifacts.load_dict(setting_path)

        # Get epochNum with Best Validation Score
        run_id = result_df.get_column("run_id")[0]
        self.epochNum = len(mlflow.tracking.MlflowClient().get_metric_history(run_id, "VALID_LOSS")) - 10 - 1
        
        self.hyparams = hyparams

        return self

    def __init__(self, modelName, datasetName, seed, memo=None):

        """
            func: training and testing of specified model on a dataset
            args:
                modelName: name of model
                datasetName: name of dataset
                seed: seed of optune and model initializer
        """

        # Set args.
        self.modelName = modelName
        self.datasetName = datasetName
        self.seed = seed
        self.memo = memo

        # Setup MLflow
        self.readyMLflow()
        
        # Download Best Setting on Validation
        self.dowmload_best_setting()

        # Load Dataset for Training
        DATA = getDataset(self.datasetName, self.seed, "train")

        # Generate Matrices
        shape = (DATA["user_num"], DATA["item_num"])
        data = DATA["df_TRAIN"]["rating"].to_numpy()
        row = DATA["df_TRAIN"]["user_id"].to_numpy()
        col = DATA["df_TRAIN"]["item_id"].to_numpy()
        self.X = csr_array((data, (row, col)), shape=shape).toarray()
        self.W = np.where(self.X == 0, 0., 1.)
        data = DATA["df_EVALUATION"]["rating"].to_numpy()
        row = DATA["df_EVALUATION"]["user_id"].to_numpy()
        col = DATA["df_EVALUATION"]["item_id"].to_numpy()
        self.X_VALID = csr_array((data, (row, col)), shape=shape).toarray()
        self.W_VALID = np.where(self.X_VALID == 0, 0., 1.)

        # Calc. Test Score
        self.evaluate()

modelName = args.model
for datasetName in args.dataset:
    for seed in args.seed:
        test( modelName, datasetName, seed, memo=args.memo )
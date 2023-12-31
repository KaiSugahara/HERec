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

class train:
    
    def objective(self, trial):

        with mlflow.start_run(experiment_id=self.experiment_id) as run:

            # Get Hyper-parameter Setting
            hyparams = self.suggester.suggest_hyparam(trial)

            self.seed

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
                    run = run,
                )
            elif self.modelName == "IHSR":
                model = IHSR(
                    d = hyparams["model"]["d"],
                    n_by_level = {**{1: hyparams["model"]["userClusterNum"]}, **{l+1: max(math.ceil(hyparams["model"]["userClusterNum"] / (2**l)), 1) for l in range(1, hyparams["model"]["userClusterDepth"])}},
                    m_by_level = {**{1: hyparams["model"]["itemClusterNum"]}, **{l+1: max(math.ceil(hyparams["model"]["itemClusterNum"] / (2**l)), 1) for l in range(1, hyparams["model"]["itemClusterDepth"])}},
                    lam = hyparams["model"]["lam"],
                    seed = self.seed,
                    run = run,
                )

            # Train
            model.fit(self.X, self.W, self.X_VALID, self.W_VALID)
            print()

        return model.best_valid_loss

    def readyMLflow(self):

        load_dotenv(".env")

        EXPERIMENT_NAME = f"{self.datasetName}-{self.modelName}-TRAIN"
        if (experiment := mlflow.get_experiment_by_name(EXPERIMENT_NAME)) is None:
            self.experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
        else:
            self.experiment_id = experiment.experiment_id

        print("Experiment Name:", EXPERIMENT_NAME)
        print("Experiment ID:", self.experiment_id)

        return self

    def __init__(self, modelName, datasetName, suggester, seed, memo=None):

        """
            func: training and testing of specified model on a dataset
            args:
                modelName: name of model
                datasetName: name of dataset
                suggester: suggester of hyperparameter
                seed: seed of optune and model initializer
        """

        # Set args.
        self.modelName = modelName
        self.datasetName = datasetName
        self.suggester = suggester
        self.seed = seed
        self.memo = memo

        # Setup MLflow
        self.readyMLflow()

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

        # TPE
        study = optuna.create_study( sampler=optuna.samplers.TPESampler(seed=self.seed) )
        study.optimize( self.objective, n_trials=100 )

modelName, suggester = "eTREE", hyParamSuggester(["../setting/model/eTREE.yaml"])
for datasetName in ["ML100K", "ML1M", "Ciao_PART", "Ciao", "Yelp"]:
    for seed in range(3):
        train( modelName, datasetName, suggester, seed, memo="des" )

# modelName, suggester = "IHSR", hyParamSuggester(["../setting/model/IHSR.yaml"])
# for datasetName in ["ML100K", "ML1M", "Ciao_PART", "Ciao", "Yelp"]:
#     for seed in range(3):
#         train( modelName, datasetName, suggester, seed, memo="des" )
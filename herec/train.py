from dotenv import load_dotenv
import mlflow
import optuna
from herec.reader import *
from herec.utils import *

class train:
    
    def objective(self, trial):

        with mlflow.start_run(experiment_id=self.experiment_id) as run:

            # Get Hyper-parameter Setting
            hyparams = self.suggester.suggest_hyparam(trial)

            # Set Trainer Seed same as Sampler Seed
            hyparams["trainer"]["seed"] = self.seed

            # Save Hyper-parameter to MLFlow
            mlflow.log_params(hyparams["model"])
            mlflow.log_params(hyparams["trainer"])
            if "loader" in hyparams.keys():
                mlflow.log_params(hyparams["loader"])
            mlflow.log_dict(hyparams, "params.json")

            # Save memo to MLFlow
            if self.memo is not None:
                mlflow.set_tag('memo', self.memo)
        
            # Define Model
            model = getModel(self.modelName, hyparams, self.DATA)

            # Set Loader
            self.targetLoader = getLoader(self.modelName, hyparams)

            # Train
            trainer = self.targetTrainer(model=model, dataLoader=self.targetLoader, run=run, ckpt_dir=f"{getRepositoryPath()}/checkpoint/", **hyparams["trainer"])
            trainer.fit(self.DATA["df_TRAIN"], self.DATA["df_EVALUATION"])
            trainer.clear_cache()
            print()
        
            # Get Best Validation Loss
            best_valid_loss = trainer.get_best_score()

            # Save Best Valid. Loss to MLFlow
            mlflow.log_metric("BEST_VALID_LOSS", best_valid_loss)

        return best_valid_loss

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

        # Set helper(s)
        self.targetTrainer = getTrainer(self.modelName)

        # Setup MLflow
        self.readyMLflow()

        # Load Dataset for Training
        self.DATA = getDataset(self.datasetName, self.seed, "train")

        # TPE
        study = optuna.create_study( sampler=optuna.samplers.TPESampler(seed=self.seed) )
        study.optimize( self.objective, n_trials=100 )
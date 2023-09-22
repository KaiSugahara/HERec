from dotenv import load_dotenv
import mlflow
import math
import optuna
from herec.reader import *
from herec.utils import getRepositoryPath

class train:

    def set_helper(self):

        if self.model_name in ["MF", "HE_MF", "FM", "HE_FM"]:

            # DataLoader
            from herec.loader import ratingLoader as targetLoader
            # Trainer
            from herec.trainer import ratingTrainer as targetTrainer

        else:
            
            raise Exception()
        
        self.targetLoader = targetLoader
        self.targetTrainer = targetTrainer

    def generate_model(self, hyparam):

        """
            func: generate a model from model class
        """

        if self.model_name == "MF":
            
            from herec.model import MF
            return MF(
                user_num=self.DATA["user_num"],
                item_num=self.DATA["item_num"],
                **hyparam["model"]
            )

        if self.model_name == "HE_MF":

            from herec.model import HE_MF
            return HE_MF(
                user_num=self.DATA["user_num"],
                item_num=self.DATA["item_num"],
                userClusterNums=[max(math.ceil(self.DATA["user_num"] * hyparam["model"].pop("userClusterNumProps")), 1)],
                itemClusterNums=[max(math.ceil(self.DATA["item_num"] * hyparam["model"].pop("itemClusterNumProps")), 1)],
                **hyparam["model"]
            )
        
        if self.model_name == "FM":

            from herec.model import FM
            return FM(
                user_num=self.DATA["user_num"],
                item_num=self.DATA["item_num"],
                **hyparam["model"]
            )

        if self.model_name == "HE_FM":

            from herec.model import HE_FM
            return HE_FM(
                user_num=self.DATA["user_num"],
                item_num=self.DATA["item_num"],
                userClusterNums=[max(math.ceil(self.DATA["user_num"] * hyparam["model"].pop("userClusterNumProps")), 1)],
                itemClusterNums=[max(math.ceil(self.DATA["item_num"] * hyparam["model"].pop("itemClusterNumProps")), 1)],
                **hyparam["model"]
            )
        
        raise Exception()
    
    def objective(self, trial):

        with mlflow.start_run(experiment_id=self.experiment_id) as run:

            # Get Hyper-parameter Setting
            hyparam = self.suggester.suggest_hyparam(trial)

            # Set Trainer Seed same as Sampler Seed
            hyparam["trainer"]["seed"] = self.seed

            # Save Hyper-parameter to MLFlow
            mlflow.log_params(hyparam["model"])
            mlflow.log_params(hyparam["trainer"])
            mlflow.log_dict(hyparam, "params.json")

            # Save memo to MLFlow
            if self.memo is not None:
                mlflow.set_tag('memo', self.memo)
        
            # Define Model
            model = self.generate_model(hyparam)

            # Train
            trainer = self.targetTrainer(model=model, dataLoader=self.targetLoader, run=run, ckpt_dir=f"{getRepositoryPath()}/checkpoint/", verbose=1, **hyparam["trainer"])
            trainer.fit(self.DATA["df_TRAIN"], self.DATA["df_VALID"])
            trainer.clear_cache()
            print()
        
            # Get Best Validation Loss
            best_valid_loss = trainer.score( trainer.get_best_params(), self.DATA["df_VALID"] )

            # Save Best Valid. Loss to MLFlow
            mlflow.log_metric("BEST_VALID_LOSS", best_valid_loss)

        return best_valid_loss
    
    def load_dataset(self):

        if self.dataset_name == "ML100K":
            reader = ML100K()
        elif self.dataset_name == "ML1M":
            reader = ML1M()
        elif self.dataset_name == "ML10M":
            reader = ML10M()
        elif self.dataset_name == "ML25M":
            reader = ML25M()
        elif self.dataset_name == "Ciao":
            reader = Ciao()
        elif self.dataset_name == "Ciao_PART":
            reader = Ciao_PART()

        self.DATA = reader.VALIDATION[self.seed].copy()

        # Print Statistics
        print("shape of df_TRAIN:", self.DATA["df_TRAIN"].shape)
        print("shape of df_VALID:", self.DATA["df_VALID"].shape)
        if "user_num" in self.DATA.keys():
            print("User #:", self.DATA["user_num"])
        if "item_num" in self.DATA.keys():
            print("Item #:", self.DATA["item_num"])
        
        return self

    def setup_mlflow(self):

        load_dotenv(".env")

        EXPERIMENT_NAME = f"HeRec-{self.model_name}-{self.dataset_name}"
        if (experiment := mlflow.get_experiment_by_name(EXPERIMENT_NAME)) is None:
            self.experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
        else:
            self.experiment_id = experiment.experiment_id

        print("実験名:", EXPERIMENT_NAME)
        print("実験ID:", self.experiment_id)

        return self

    def __init__(self, model_name, dataset_name, suggester, seed, memo=None):

        """
            func: training and testing of specified model on a dataset
            args:
                model_name: name of model
                dataset_name: name of dataset
                suggester: suggester of hyperparameter
                seed: seed of optune and model initializer
        """

        # Set args.
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.suggester = suggester
        self.seed = seed
        self.memo = memo

        # Set helper(s)
        self.set_helper()

        # Setup MLflow
        self.setup_mlflow()

        # Load Dataset for Validation
        self.load_dataset()

        # TPE
        study = optuna.create_study( sampler=optuna.samplers.TPESampler(seed=self.seed) )
        study.optimize(self.objective, n_trials=100)
import mlflow
import optuna

class train_test:

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
                user_num=self.reader.user_num,
                item_num=self.reader.item_num,
                **hyparam["model"]
            )

        if self.model_name == "HE_MF":

            from herec.model import HE_MF
            return HE_MF(
                user_num=self.reader.user_num,
                item_num=self.reader.item_num,
                clusterNums=[hyparam["model"].pop("clusterNums")],
                **hyparam["model"]
            )
        
        if self.model_name == "FM":

            from herec.model import FM
            return FM(
                user_num=self.reader.user_num,
                item_num=self.reader.item_num,
                **hyparam["model"]
            )

        if self.model_name == "HE_FM":

            from herec.model import HE_FM
            return HE_FM(
                user_num=self.reader.user_num,
                item_num=self.reader.item_num,
                clusterNums=[hyparam["model"].pop("clusterNums")],
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
        
            # Define Model
            model = self.generate_model(hyparam)

            # Train
            trainer = self.targetTrainer(model=model, dataLoader=self.targetLoader, run=run, ckpt_dir="../checkpoint/", verbose=1, **hyparam["trainer"])
            trainer.fit(self.reader.df_SUBSET["TRAIN"], self.reader.df_SUBSET["VALID"])
            trainer.clear_cache()
        
            # Get Best Validation Loss
            best_valid_loss = trainer.score( trainer.get_best_params(), self.reader.df_SUBSET["VALID"] )

            # Save Best Valid. Loss to MLFlow
            mlflow.log_metric("BEST_VALID_LOSS", best_valid_loss)

            # Get Test Loss
            test_loss = trainer.score( trainer.get_best_params(), self.reader.df_SUBSET["TEST"] )

            # Save Test Loss to MLFlow
            mlflow.log_metric("TEST_LOSS", test_loss)

        return best_valid_loss

    def __init__(self, model_name, reader, suggester, seed, experiment_id):

        """
            func: training and testing of specified model on a dataset
            args:
                model_name: name of model
                reader: reader of dataset
                suggester: suggester of hyperparameter
                seed: seed of optune and model initializer
                experiment_id: experiment id of MLflow
        """

        # Set args.
        self.model_name = model_name
        self.reader = reader
        self.suggester = suggester
        self.seed = seed
        self.experiment_id = experiment_id

        # Set helper(s)
        self.set_helper()

        # TPE
        study = optuna.create_study( sampler=optuna.samplers.TPESampler(seed=self.seed) )
        study.optimize(self.objective, n_trials=100)
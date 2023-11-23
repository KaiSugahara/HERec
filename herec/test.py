from dotenv import load_dotenv
import mlflow
from herec.utils import *

class test:
    
    def evaluate(self):

        with mlflow.start_run(experiment_id=self.experiment_id) as run:

            # Save Hyper-parameter to MLFlow
            mlflow.log_params(self.hyparams["model"])
            mlflow.log_params(self.hyparams["trainer"])
            if "loader" in self.hyparams.keys():
                mlflow.log_params(self.hyparams["loader"])
            mlflow.log_dict(self.hyparams, "params.json")

            # Save memo to MLFlow
            if self.memo is not None:
                mlflow.set_tag('memo', self.memo)
        
            # Define Model
            model = getModel(self.modelName, self.hyparams, self.DATA)

            # Set Loader
            self.targetLoader = getLoader(self.modelName, self.hyparams)

            # Train
            trainer = self.targetTrainer(model=model, dataLoader=self.targetLoader, run=run, ckpt_dir=f"{getRepositoryPath()}/checkpoint/", **self.hyparams["trainer"])
            trainer.fit(self.DATA["df_TRAIN"], self.DATA["df_EVALUATION"])
            trainer.clear_cache()
            print()

        return self

    def readyMLflow(self):

        load_dotenv(".env")

        EXPERIMENT_NAME = f"HeRec-TEST-{self.datasetName}-{self.modelName}"
        if (experiment := mlflow.get_experiment_by_name(EXPERIMENT_NAME)) is None:
            self.experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
        else:
            self.experiment_id = experiment.experiment_id

        print("Experiment Name:", EXPERIMENT_NAME)
        print("Experiment ID:", self.experiment_id)

        return self

    def dowmload_best_setting(self):

        # Get Validation Results
        result_df = resultLoader( f"HeRec-TRAIN-{self.datasetName}-{self.modelName}" ).get_results_by_fold( self.seed )
        
        # Check
        if result_df.height != 100:
            raise Exception("Please evaluate test set after valiadation")

        # Get Hyparam. Setting with Best Validation Score
        setting_path = result_df.get_column("artifact_uri")[0] + "/params.json"
        hyparams = mlflow.artifacts.load_dict(setting_path)

        # Get epochNum with Best Validation Score
        run_id = result_df.get_column("run_id")[0]
        epochNum = len(mlflow.tracking.MlflowClient().get_metric_history(run_id, "VALID_LOSS")) - hyparams["trainer"]["es_patience"] - 1
        
        # Edit Setting to disable Early Stopping
        hyparams["trainer"]["epochNum"] = epochNum
        hyparams["trainer"]["es_patience"] = 0
        
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

        # Set helper(s)
        self.targetTrainer = getTrainer(self.modelName)

        # Setup MLflow
        self.readyMLflow()

        # Download Best Setting on Validation
        self.dowmload_best_setting()

        # Load Dataset for Training
        self.DATA = getDataset(self.datasetName, self.seed, "test")

        # Calc. Test Score
        self.evaluate()
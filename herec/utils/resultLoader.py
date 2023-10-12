import polars as pl
from tqdm import tqdm
import mlflow
from dotenv import load_dotenv
from herec.utils import *

class resultLoader:

    def get_step_wise_scores(self, run_id: int, metric_name: str):
        
        history = self.mlflow_client.get_metric_history(run_id, metric_name)
        return [step.value for step in history]

    def __setup_mlflow(self, EXPERIMENT_NAME: str):

        # Load environment variables regarding MLflow
        load_dotenv(f"{getRepositoryPath()}/.env")
        
        # Set Experiment Instance
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None: raise Exception("Non-existance Experiment")

        # Get experiment id
        self.experiment_id = experiment.experiment_id

        # Set Client of MLflow
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
        print("Experiment Name:", EXPERIMENT_NAME)
        print("Experiment ID:", self.experiment_id)

        return self

    def __download_results(self):

        # Load from MLflow
        result_df = pl.from_pandas(mlflow.search_runs(experiment_ids=[self.experiment_id]))
            
        # Get column names related to metrics
        # metric_column_names = [name for name in result_df.columns if (name.startswith("metrics.")) and (name != "metrics.BEST_VALID_LOSS")]
        
        # Get step-wise metrics
        # result_df = result_df.with_columns(
        #     pl.Series([self.get_step_wise_scores(run_id, col_name.replace("metrics.", "")) for run_id in tqdm(result_df.get_column("run_id"), desc=col_name)]).alias(col_name)
        #     for col_name in metric_column_names
        # )
    
        # Sort by BEST_VALID_LOSS
        result_df = result_df.sort("metrics.BEST_VALID_LOSS")
    
        # Set
        self.result_df = result_df
        
        return self
    
    def get_results_by_fold(self, foldId: int):

        return self.result_df.filter(
            pl.col("params.seed") == str(foldId)
        )

    def __init__(self, EXPERIMENT_NAME: str):

        self.__setup_mlflow(EXPERIMENT_NAME)
        self.__download_results()
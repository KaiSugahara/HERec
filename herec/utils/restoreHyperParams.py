import mlflow
from dotenv import load_dotenv
from herec.utils import *

def restoreHyperParams( run_id ):

    # Load environment variables regarding MLflow
    load_dotenv(f"{getRepositoryPath()}/.env")
    
    return mlflow.artifacts.load_dict( f"{mlflow.get_run(run_id).info.artifact_uri}/params.json" )
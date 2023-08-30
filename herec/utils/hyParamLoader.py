import yaml
import itertools

def hyParamLoader(yaml_path):

    # Read
    with open(yaml_path, "r") as f:
        hyParamSetting = yaml.safe_load(f.read())

    # Extract Combinations
    hyParamList = [dict(zip(hyParamSetting.keys(), x)) for x in itertools.product(*hyParamSetting.values())]

    return hyParamList
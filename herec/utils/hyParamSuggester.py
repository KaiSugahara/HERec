import yaml

class hyParamSuggester():

    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            self.rawSetting = yaml.safe_load(f.read())

    def set_suggest_method(self, name, conf, trial):
        if conf["type"] == "categorical":
            return trial.suggest_categorical(name, conf["choices"])
        if conf["type"] == "float":
            return trial.suggest_float(name, low=float(conf["low"]), high=float(conf["high"]), log=conf["log"])
        raise Exception("")

    def suggest_hyparam(self, trial):
        return dict(
            model = {name: self.set_suggest_method(name, conf, trial) for name, conf in self.rawSetting["model"].items()},
            trainer = {name: self.set_suggest_method(name, conf, trial) for name, conf in self.rawSetting["trainer"].items()},
        )
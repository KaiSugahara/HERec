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
        if conf["type"] == "int":
            return trial.suggest_int(name, low=int(conf["low"]), high=int(conf["high"]), log=conf["log"])
        raise Exception("Invalid Type")

    def suggest_hyparam(self, trial):

        hyparam = {}

        if "model" in self.rawSetting.keys():
            hyparam["model"] = {name: self.set_suggest_method(name, conf, trial) for name, conf in self.rawSetting["model"].items()}

        if "trainer" in self.rawSetting.keys():
            hyparam["trainer"] = {name: self.set_suggest_method(name, conf, trial) for name, conf in self.rawSetting["trainer"].items()}

        if "loader" in self.rawSetting.keys():
            hyparam["loader"] = {name: self.set_suggest_method(name, conf, trial) for name, conf in self.rawSetting["loader"].items()}

        return hyparam
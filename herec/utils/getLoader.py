def getLoader(modelName: str, hyparams: dict):
    
    if modelName in ["MF", "HE_MF", "FM", "HE_FM"]:

        from ..loader import ratingLoader as targetLoader

    elif modelName in ["MF_BPR", "HE_MF_BPR"]:

        from ..loader import bprLoader as targetLoader
        
    elif modelName in ["MF_BCE", "HE_MF_BCE"]:

        from ..loader import bceLoader as targetLoader
        
    elif modelName in ["MF_SSM", "HE_MF_SSM"]:

        from ..loader import ssmLoader as targetLoader
        targetLoader.n_neg = hyparams["loader"].pop("n_neg")
        
    elif modelName in ["GRU4Rec", "HE_GRU4Rec"]:

        from ..loader import sessionLoader as targetLoader

    else:
        
        raise Exception()

    return targetLoader
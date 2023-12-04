def getLoader(modelName: str, hyparams: dict):
    
    if modelName in ["MF", "HE_MF", "HSE_MF", "FM", "HE_FM", "HSE_FM"]:

        from ..loader import ratingLoader as targetLoader

    elif modelName in ["MF_BPR", "HE_MF_BPR", "HSE_MF_BPR", "MF_BCE", "HE_MF_BCE", "HSE_MF_BCE", "MF_SSM", "HE_MF_SSM", "HSE_MF_SSM"]:

        from ..loader import implicitLoader as targetLoader
        targetLoader.n_neg = hyparams["loader"].pop("n_neg")
        targetLoader.has_weight = hyparams["loader"].pop("has_weight")
        targetLoader.sampler = hyparams["loader"].pop("sampler")
        
    elif modelName in ["GRU4Rec", "HE_GRU4Rec", "HSE_GRU4Rec"]:

        from ..loader import sessionLoader as targetLoader

    else:
        
        raise Exception()

    return targetLoader
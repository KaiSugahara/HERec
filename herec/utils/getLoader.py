def getLoader(modelName: str, hyparams: dict):

    # Explicit RS    
    if modelName in [
        "MF", "HE_MF",
        "FM", "HE_FM",
        "NeuMF", "HE_NeuMF",
    ]:

        from ..loader import ratingLoader as targetLoader

    # Implicit RS
    elif modelName in [
        "MF_BPR", "HE_MF_BPR",
        "HE_MF_USER_BPR",
        "HE_MF_ITEM_BPR",
        "NeuMF_BPR", "HE_NeuMF_BPR",
        "ProtoMF_BPR",
    ]:

        from ..loader import implicitLoader as targetLoader
        targetLoader.n_neg = hyparams["loader"].pop("n_neg")

    else:
        
        raise Exception()

    return targetLoader
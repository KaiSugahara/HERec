def getLoader(modelName: str, hyparams: dict):

    # Explicit RS    
    if modelName in [
        "MF", "HE_MF", "HSE_MF", "DHE_MF",
        "FM", "HE_FM", "HSE_FM", "DHE_FM",
        "NeuMF", "HE_NeuMF", "HSE_NeuMF", "DHE_NeuMF",
    ]:

        from ..loader import ratingLoader as targetLoader

    # Implicit RS
    elif modelName in [
        "MF_BPR", "HE_MF_BPR", "HSE_MF_BPR", "DHE_MF_BPR",
        "HE_MF_USER_BPR",
        "HE_MF_ITEM_BPR",
        "NeuMF_BPR", "HE_NeuMF_BPR", "HSE_NeuMF_BPR", "DHE_NeuMF_BPR",
        "MF_BCE", "HE_MF_BCE", "HSE_MF_BCE", "DHE_MF_BCE",
        "MF_SSM", "HE_MF_SSM", "HSE_MF_SSM", "DHE_MF_SSM",
    ]:

        from ..loader import implicitLoader as targetLoader
        targetLoader.n_neg = hyparams["loader"].pop("n_neg")

    # Sequential RS
    elif modelName in [
        "GRU4Rec", "HE_GRU4Rec", "HSE_GRU4Rec", "DHE_GRU4Rec",
    ]:

        from ..loader import sessionLoader as targetLoader

    else:
        
        raise Exception()

    return targetLoader
def getTrainer(modelName: str):
    
    # Explicit RS
    if modelName in [
        "MF", "HE_MF",
        "FM", "HE_FM",
        "NeuMF", "HE_NeuMF",
    ]:

        from herec.trainer import ratingTrainer as targetTrainer

    # Implicit RS (BPR)
    elif modelName in [
        "MF_BPR", "HE_MF_BPR",
        "HE_MF_USER_BPR",
        "HE_MF_ITEM_BPR",
        "NeuMF_BPR", "HE_NeuMF_BPR",
        "ProtoMF_BPR",
    ]:

        from herec.trainer import bprTrainer as targetTrainer
    
    # Sequential RS
    elif modelName in [
        "GRU4Rec", "HE_GRU4Rec",
    ]:

        from herec.trainer import sessionTrainer as targetTrainer

    else:
        
        raise Exception()

    return targetTrainer
def getTrainer(modelName: str):
    
    # Explicit RS
    if modelName in [
        "MF", "HE_MF", "HSE_MF",
        "FM", "HE_FM", "HSE_FM",
        "NeuMF", "HE_NeuMF", "HSE_NeuMF",
    ]:

        from herec.trainer import ratingTrainer as targetTrainer

    # Implicit RS (BPR)
    elif modelName in [
        "MF_BPR", "HE_MF_BPR", "HSE_MF_BPR",
        "HE_MF_USER_BPR",
        "HE_MF_ITEM_BPR",
    ]:

        from herec.trainer import bprTrainer as targetTrainer
    
    # Implicit RS (BCE)
    elif modelName in [
        "MF_BCE", "HE_MF_BCE", "HSE_MF_BCE",
    ]:

        from herec.trainer import bceTrainer as targetTrainer
    
    # Implicit RS (SSM)
    elif modelName in [
        "MF_SSM", "HE_MF_SSM", "HSE_MF_SSM",
    ]:

        from herec.trainer import ssmTrainer as targetTrainer
    
    # Sequential RS
    elif modelName in [
        "GRU4Rec", "HE_GRU4Rec", "HSE_GRU4Rec"
    ]:

        from herec.trainer import sessionTrainer as targetTrainer

    else:
        
        raise Exception()

    return targetTrainer
def getTrainer(modelName: str):
    
    if modelName in ["MF", "HE_MF", "HSE_MF", "FM", "HE_FM", "HSE_FM"]:

        from herec.trainer import ratingTrainer as targetTrainer

    elif modelName in ["MF_BPR", "HE_MF_BPR", "HSE_MF_BPR"]:

        from herec.trainer import bprTrainer as targetTrainer
        
    elif modelName in ["MF_BCE", "HE_MF_BCE", "HSE_MF_BCE"]:

        from herec.trainer import bceTrainer as targetTrainer
        
    elif modelName in ["MF_SSM", "HE_MF_SSM", "HSE_MF_SSM"]:

        from herec.trainer import ssmTrainer as targetTrainer
        
    elif modelName in ["GRU4Rec", "HE_GRU4Rec", "HSE_GRU4Rec"]:

        from herec.trainer import sessionTrainer as targetTrainer

    else:
        
        raise Exception()

    return targetTrainer
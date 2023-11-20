def getTrainer(modelName: str):
    
    if modelName in ["MF", "HE_MF", "FM", "HE_FM"]:

        from herec.trainer import ratingTrainer as targetTrainer

    elif modelName in ["MF_BPR", "HE_MF_BPR"]:

        from herec.trainer import bprTrainer as targetTrainer
        
    elif modelName in ["MF_SSM", "HE_MF_SSM"]:

        from herec.trainer import ssmTrainer as targetTrainer
        
    elif modelName in ["GRU4Rec", "HE_GRU4Rec"]:

        from herec.trainer import sessionTrainer as targetTrainer

    else:
        
        raise Exception()

    return targetTrainer
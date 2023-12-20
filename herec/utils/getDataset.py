from ..reader import *
from . import *

def getDataset(datasetName: str, seed: int, evalStage: str):

    if datasetName == "ML100K":
        reader = ML100K()
    elif datasetName == "ML100K_IMPLICIT":
        reader = ML100K_IMPLICIT()
    elif datasetName == "ML1M":
        reader = ML1M()
    elif datasetName == "ML1M_IMPLICIT":
        reader = ML1M_IMPLICIT()
    elif datasetName == "ML10M":
        reader = ML10M()
    elif datasetName == "ML25M":
        reader = ML25M()
    elif datasetName == "Ciao":
        reader = Ciao()
    elif datasetName == "Ciao_PART":
        reader = Ciao_PART()
    elif datasetName == "Yelp":
        reader = Yelp()
    elif datasetName == "Twitch100K":
        reader = Twitch100K()
    elif datasetName == "DIGINETICA":
        reader = DIGINETICA()
    elif datasetName == "Pinterest":
        reader = Pinterest()
    elif datasetName == "AMAZON_M2":
        reader = AMAZON_M2()
    elif datasetName == "G1NEWS_SESSION":
        reader = G1NEWS_SESSION()

    DATA = reader.get(seed, evalStage).copy()

    # Print Statistics
    print("shape of df_TRAIN:", DATA["df_TRAIN"].shape)
    if type(DATA["df_EVALUATION"]) != dict:
        print("shape of df_EVALUATION:", DATA["df_EVALUATION"].shape)
    if "user_num" in DATA.keys():
        print("User #:", DATA["user_num"])
    if "item_num" in DATA.keys():
        print("Item #:", DATA["item_num"])

    return DATA
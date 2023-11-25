import math
from ..model import *

def getModel(modelName: str, hyparams: dict, DATA: dict):
    
    if modelName in ["MF", "MF_BPR", "MF_BCE", "MF_SSM"]:
            
        return MF(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
            **hyparams["model"]
        )

    if modelName in ["HE_MF", "HE_MF_BPR", "HE_MF_BCE", "HE_MF_SSM"]:

        return HE_MF(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
            userClusterNums=[num := hyparams["model"].pop("userClusterNum")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("userHierarchyDepth"))],
            itemClusterNums=[num := hyparams["model"].pop("itemClusterNum")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("itemHierarchyDepth"))],
            **hyparams["model"],
        )
    
    if modelName == "FM":

        return FM(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
            **hyparams["model"]
        )

    if modelName == "HE_FM":

        return HE_FM(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
            userClusterNums=[num := hyparams["model"].pop("userClusterNum")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("userHierarchyDepth"))],
            itemClusterNums=[num := hyparams["model"].pop("itemClusterNum")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("itemHierarchyDepth"))],
            **hyparams["model"]
        )
        
    if modelName == "GRU4Rec":

        return GRU4Rec(
            item_num=DATA["item_num"]+1,
            GRU_LAYER_SIZES=[num := hyparams["model"].pop("gruLayerSize")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("gruLayerDepth"))],
            FF_LAYER_SIZES=[num := hyparams["model"].pop("ffLayerSize")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("ffLayerDepth"))],
            **hyparams["model"]
        )
        
    if modelName == "HE_GRU4Rec":

        return HE_GRU4Rec(
            item_num=DATA["item_num"]+1,
            itemClusterNums=[num := hyparams["model"].pop("itemClusterNum")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("itemHierarchyDepth"))],
            GRU_LAYER_SIZES=[num := hyparams["model"].pop("gruLayerSize")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("gruLayerDepth"))],
            FF_LAYER_SIZES=[num := hyparams["model"].pop("ffLayerSize")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("ffLayerDepth"))],
            **hyparams["model"]
        )
    
    raise Exception()
import math
from ..model import *

def getModel(modelName: str, hyparams: dict, DATA: dict):
    
    if modelName in ["MF", "MF_BPR"]:
            
        return MF(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
            **hyparams["model"]
        )

    if modelName in ["HE_MF", "HE_MF_BPR"]:

        return HE_MF(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
            userClusterNums=[num := hyparams["model"].pop("userClusterNum")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("userHierarchyDepth"))],
            itemClusterNums=[num := hyparams["model"].pop("itemClusterNum")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("itemHierarchyDepth"))],
            **hyparams["model"],
        )
        
    if modelName in ["HE_MF_USER_BPR"]:

        return HE_MF_USER(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
            userClusterNums=[num := hyparams["model"].pop("userClusterNum")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("userHierarchyDepth"))],
            **hyparams["model"],
        )
        
    if modelName in ["HE_MF_ITEM_BPR"]:

        return HE_MF_ITEM(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
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
        
    if modelName in ["NeuMF", "NeuMF_BPR"]:
            
        return NeuMF(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
            **hyparams["model"]
        )

    if modelName in ["HE_NeuMF", "HE_NeuMF_BPR"]:

        return HE_NeuMF(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
            userClusterNums=[num := hyparams["model"].pop("userClusterNum")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("userHierarchyDepth"))],
            itemClusterNums=[num := hyparams["model"].pop("itemClusterNum")] + [max(math.ceil(num / (2**l)), 1) for l in range(1, hyparams["model"].pop("itemHierarchyDepth"))],
            **hyparams["model"],
        )
        
    if modelName in ["ProtoMF_BPR"]:

        return ProtoMF(
            user_num=DATA["user_num"],
            item_num=DATA["item_num"],
            **hyparams["model"],
        )
    
    raise Exception()
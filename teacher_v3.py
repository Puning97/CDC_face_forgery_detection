import torch
from feature_transformer_CCL import Feature_Trans_img as create_model
from collections import OrderedDict
def id_teacher():
    checkpoint = torch.load('/data/chpt/ACCV2022/sota/weights/idteacher_66.13_1_6_1024.pth', map_location='cpu')
    
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        k_new = k[7:]
        new_state_dict[k_new] = v
    
    checkpoints = {k: v for k, v in new_state_dict.items()}
    model=create_model()
    model_dict = model.state_dict()
    model_dict.update(checkpoints)
    model.load_state_dict(model_dict)
    return model
def moion_teacher():
    checkpoint = torch.load('/data/ckpt/ACCV2022/sota/weights/motionteacher_6963_1_6_1024.pth', map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        k_new = k[7:]
        new_state_dict[k_new] = v
    
    checkpoints = {k: v for k, v in new_state_dict.items()}
    model=create_model()
    model_dict = model.state_dict()
    model_dict.update(checkpoints)
    model.load_state_dict(model_dict)
    return model

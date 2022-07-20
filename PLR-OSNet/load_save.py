import torch
import onnx
from torchreid.models.osnet import OSBlock, OSBlock, OSBlock, OSNet
def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    from collections import OrderedDict



    cached_file = "/home/indemind/Code/CLionprojects/Rubby/yolov5_Deepsort_rknn/build_deepsort/model/ckpt.pt"



    state_dict = torch.load(cached_file)
    print(state_dict)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.
        
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(cached_file))
    else:
        print('Successfully loaded imagenet pretrained weights from "{}"'.format(cached_file))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded '
                  'due to unmatched keys or layer size: {}'.format(discarded_layers))

model = OSNet(1000, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                channels=[16, 64, 96, 128], loss=None)
example = torch.rand(1,3,128,64)


init_pretrained_weights(model, key='osnet_x0_25')

torch.onnx.export(model, example, "helmet2.onnx", verbose=True, opset_version=12)

# model.eval()
# traced_script_module = torch.jit.trace(model,example)
# # 
# cccc = traced_script_module.save("./loadsave2.pt")
# torch.save(model.state_dict(),'loadsave.pt',_use_new_zipfile_serialization = True)

# model = torch.jit.load('./loadsave.pt')
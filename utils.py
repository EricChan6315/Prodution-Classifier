import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import yaml


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = DotDict(v)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(config_path):

    with open(config_path, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)

    return args

def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

def logging(epoch, train_loss, train_acc=None, valid_loss=None, valid_acc=None, outdir=None, step=None):
    log_path = os.path.join(outdir, 'log.txt')
    if step is not None and valid_loss is None and valid_acc is None:
        log_str = f"epoch: {epoch} | step: {step} | train_loss: {train_loss:.6f} | train_acc: {train_acc:.6f}\n"
    else:
        log_str = f"epoch: {epoch} | train_loss: {train_loss:.6f} | train_acc: {train_acc:.6f} | valid_loss: {valid_loss:.6f} | valid_acc: {valid_acc:.6f}\n"
    with open(log_path, 'a') as f:
        f.write(log_str)

def save_model(model, outdir=None, name='model'):
    # path
    path_params = os.path.join(outdir, name+'_params.pth')

    # check
    print(' [*] saving model to {}, name: {}'.format(outdir, name))

    # save
    torch.save(model.state_dict(), path_params)

def load_model(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(ckpt_path, msg))
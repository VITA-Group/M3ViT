def cal_flops(w, h, k, c_in, c_out):
    """ calculate the actual flops of one example

    c_in and c_out are vector across the whole batch
    """
    return w * h * k * k * c_in * c_out

def load_pretrained_v2(model, path):
    ckpt = torch.load(path)['state_dict']
    model_state = model.state_dict()

    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_state:
            new_ckpt[k] = v

    model_state.update(new_ckpt)
    model.load_state_dict(model_state)
    return model
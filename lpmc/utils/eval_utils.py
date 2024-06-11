import torch

def print_model_params(model):
    n_parameters = sum(p.numel() for p in model.parameters())
    train_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("============")
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print('number train of params (M): %.2f' % (train_n_parameters / 1.e6))
    print("============")
    
def load_pretrained(save_dir, model, mdp=False):
    pretrained_object = torch.load(f'{save_dir}/transfer.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    save_epoch = pretrained_object['epoch']
    if mdp:
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    return model, save_epoch
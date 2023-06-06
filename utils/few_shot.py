import torch


def split_shot_query(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:] # data:320(4*5*(1+15)),3,80,80
    data = data.view(ep_per_batch, way, shot + query, *img_shape) # 4,5,16,3,80,80
    x_shot, x_query = data.split([shot, query], dim=2) # 4,5,1,3,80,80; 4,5,15,3,80,80
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)#4,75,3,80,80
    return x_shot, x_query


def make_nk_label(n, k, ep_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1) # [15*0,15*1,15*2,15*3,15*4]
    label = label.repeat(ep_per_batch)
    return label # relabeled query set


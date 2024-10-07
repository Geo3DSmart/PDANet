from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize


try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        elif key in ['Sparse_lidar']:
            store_val = []
            for i_bs in val:
                store_alone = {'Sparse_lidar':i_bs}
                store_val.append(store_alone)
            get_sparse = sparse_collate_fn(store_val)

            for keykk, value in get_sparse.items():
                batch_dict[key] =  value.cuda()

        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()



def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict) # 加载数据进行GPU
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

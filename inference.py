import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *
from utils.visualization import *

from matplotlib import pyplot as plt

torch.backends.cudnn.benchmark = True


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    # args = get_opts()
    # print(args)
    w, h = 256, 256

    kwargs = {'root_dir': sys.argv[1], 'split': 'test',}
    dataset = dataset_dict['klevr'](**kwargs)
    print(dataset.white_back)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    nerf_fine = NeRF()
    load_ckpt(nerf_coarse, '../nerf.ckpt', model_name='nerf_coarse')
    load_ckpt(nerf_fine, '../nerf.ckpt', model_name='nerf_fine')
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()

    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    psnrs = []
    
    output_path = sys.argv[2]
    dir_name = os.path.dirname(output_path)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    # os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    64, 64, False,
                                    32*1024*4,
                                    dataset.white_back)
        
        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
        
        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(output_path, f"{sample['id']:05d}.png"), img_pred_)
    
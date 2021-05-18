from mmdet3d.ops.furthest_point_sample import furthest_point_sample
import torch

if __name__ == '__main__':
    x = torch.randn((2, 16384, 3))
    x = x.cuda()
    idx = furthest_point_sample(x, 4096)

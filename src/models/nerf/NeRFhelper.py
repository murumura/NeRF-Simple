import collections
import numpy as np
import torch
import cv2
import torchvision.transforms as T
from PIL import Image

def intervals_to_ray_points(point_intervals, ray_directions, ray_origin):
    """Through depths of the sampled positions('point_intervals') and the starting point('ray_origin') 
        and direction vector of the light('ray_directions'), calculate each point on the light
    Args:
        point_intervals (torch.tensor): (ray_count, num_samples) : Depths of the sampled positions along the ray
        ray_directions (torch.tensor): (ray_count, 3)
        ray_origin (torch.tensor): (ray_count, 3)
    Return:
        ray_points(torch.tensor): Samples points along each ray: (ray_count, num_samples, 3)
    """
    ray_points = ray_origin[..., None, :] + ray_directions[..., None, :] * point_intervals[..., :, None]

    return ray_points

def cast_to_pil_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu().float()))
    return img

def cast_to_image(tensor):
    # Extract the PIL Image (output shape: (H, W, 3))
    img = cast_to_pil_image(tensor)

    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img

def cast_to_disparity_image(tensor, white_background = False):
    """
    depth: (H, W)
    """
    img = tensor.cpu()
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = (img.clamp(0., 1.) * 255).byte()

    if white_background:
        # Apply white background
        img[img == 0] = 255
     
    return img

def cast_to_depth_image(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_
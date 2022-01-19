import numpy as np
import torch
import cv2
import torchvision
import torchvision.transforms as T
from PIL import Image


def save_img(imgs, outfile):
    imgs = imgs / 2 + 0.5
    imgs = torchvision.utils.make_grid(imgs)
    torchvision.utils.save_image(imgs.clone(), outfile, nrow=8)


def psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse == 0:
        mse = 1e-5
    return -10.0 * torch.log10(torch.tensor(mse, dtype=torch.float32))


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


def cast_to_disparity_image(tensor, white_background=False):
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
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_

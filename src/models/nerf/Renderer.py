import torch

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)

    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)

    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod


class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        train_radiance_field_noise_std = 0.0,
        val_radiance_field_noise_std = 0.0,
        white_background = False,
        attenuation_threshold=1e-3,
        device = 'cuda'
    ):
        super(VolumeRenderer, self).__init__()
        self.train_radiance_field_noise_std = train_radiance_field_noise_std
        self.val_radiance_field_noise_std = val_radiance_field_noise_std
        self.attenuation_threshold = attenuation_threshold
        self.use_white_bkg = white_background
        epsilon = torch.tensor([1e10]).to(device)
        epsilon.requires_grad = False
        self.register_buffer("epsilon", epsilon)

    def forward(self, radiance_field, depth_values, ray_directions):
        """
        Perform volume rendering by computing the output of NeRF model ('radiance_field') 
        Inputs:
            radiance_field: (torch.tensor): (ray_count, num_samples, 4), rgb and sigma : output from NeRF model
            depth_values:  (torch.tensor): (ray_count, num_samples) : Depths of the sampled positions along the ray
            ray_directions: (torch.tensor): (ray_count, num_samples, 3) 
        Outputs:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
            disp_map: [num_rays]. Disparity map. 1 / depth.
            weights: [num_rays, num_samples] Weights assigned to each sampled color.
            acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
            depth_map: [num_rays]. Estimated distance to object.
            weights_mask: [num_rays, num_samples]: Record all weights whose penetrability is higher than threshold
        """
        if self.training:
            radiance_field_noise_std = self.train_radiance_field_noise_std
        else:
            radiance_field_noise_std = self.val_radiance_field_noise_std
        # distance between adjacent samples points.
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                self.epsilon.expand(depth_values[..., :1].shape),
            ),
            dim = -1,
        ) # (N_rays, N_samples_)
        # Multiply each distance by the norm (2-norm) of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * ray_directions[..., None, :].norm(p = 2, dim = -1) # (N_rays, N_samples_)

        rgb = radiance_field[..., :3]
        noise = 0.0
        if radiance_field_noise_std > 0.0:
            noise = (
                    torch.randn(
                        radiance_field[..., 3].shape,
                        dtype = radiance_field.dtype,
                        device = radiance_field.device
                    )
                    * radiance_field_noise_std
            ) #(ray_count, num_samples)
        # as suggested in https://github.com/kwea123/nerf_pl/issues/51 to see if it work fine
        sigmas = torch.nn.functional.softplus(radiance_field[..., 3] + noise)
        #sigmas = torch.nn.functional.relu(radiance_field[..., 3] + noise)
        alpha = 1.0 - torch.exp(-sigmas * deltas)
        # T_i = cumulative-product(j=1,j=(i-1))(1 - alpha_j)
        T_i = cumprod_exclusive(1.0 - alpha + 1e-10) #(ray_count, num_samples)
        # weight_i = T_i * alpha_i
        weights = alpha * T_i #(ray_count, num_samples)
        mask_weights = (T_i > self.attenuation_threshold).float() #(ray_count, num_samples)

        rgb_map = weights[..., None] * rgb #(ray_count, num_samples, 3)
        # sum over rgb value of each sample points (sum_over R,sum_over G, sum_over B) along the ray
        rgb_map = rgb_map.sum(dim = -2) #(ray_count, 3)
        # sum over weight of each ray
        acc_map = weights.sum(dim = -1) #(ray_count, )

        depth_map = (weights * depth_values).sum(dim = -1) # #(ray_count, )
        disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map) #(ray_count, )
        # filter out nan depth value
        disp_map[torch.isnan(disp_map)] = 0 

        if self.use_white_bkg:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret = {
            'rgb_map' : rgb_map, 
            'depth_map' : depth_map, 
            'weights' : weights, 
            'mask_weights':mask_weights, 
            'acc_map':acc_map, 
            'disp_map':disp_map
        }
        return ret
import torch


def intervals_to_ray_points(point_intervals, ray_directions, ray_origins):
    """Through depths of the sampled positions('point_intervals') and the starting point('ray_origin') 
        and direction vector of the light('ray_directions'), calculate each point on the light
    Args:
        point_intervals (torch.tensor): (ray_count, num_samples) : Depths of the sampled positions along the ray
        ray_directions (torch.tensor): (ray_count, 3)
        ray_origin (torch.tensor): (ray_count, 3)
    Return:
        ray_points(torch.tensor): Samples points along each ray: (ray_count, num_samples, 3)
    """
    ray_points = ray_origins[..., None, :] + ray_directions[..., None, :] * point_intervals[..., :, None]

    return ray_points


class RaySampleInterval(torch.nn.Module):
    """Defines a function sample ray interval
    Args:
        num_samples (int): Number of depth samples per ray for the coarse/fine network.
    """
    def __init__(self, num_samples, device):
        super(RaySampleInterval, self).__init__()
        self.num_samples = num_samples

        # Ray sample count
        point_intervals = torch.linspace(0.0, 1.0, self.num_samples, requires_grad=False, device=device)[None, :]
        self.register_buffer("point_intervals", point_intervals, persistent=False)

    def forward(self, ray_count, near, far, lindisp: bool = False, perturb: bool = True):
        """
        Inputs:
            ray_count: int, number of rays in input ray chunk
            near: float or array of shape [BatchSize]. Nearest distance for a ray.
            far: float or array of shape [BatchSize]. Farthest distance for a ray.
            lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
            perturb: bool, If True, each ray is sampled at stratified random points in time.
        Outputs:
            point_intervals: (ray_count, self.num_samples) : depths of the sampled points along the ray
        """
        if not hasattr(near, 'shape') and isinstance(near, float):
            near, far = near * torch.ones_like(torch.empty(ray_count, 1)), far * torch.ones_like(torch.empty(ray_count, 1))
        elif len(near.shape) > 0 and near.shape[0] == ray_count:
            near, far = near[:, None], far[:, None]
        # Sample in disparity space, as opposed to in depth space. Sampling in disparity is
        # nonlinear when viewed as depth sampling! (The closer to the camera the more samples)
        if not lindisp:
            point_intervals = near * (1.0 - self.point_intervals) + far * self.point_intervals
        else:
            point_intervals = 1.0 / (1.0 / near * (1.0 - self.point_intervals) + 1.0 / far * self.point_intervals)
        if len(near.shape) == 0 or near.shape[0] != ray_count:
            point_intervals = point_intervals.expand([ray_count, self.num_samples])

        if perturb:
            # Get intervals between samples.
            mids = 0.5 * (point_intervals[..., 1:] + point_intervals[..., :-1])
            upper = torch.cat((mids, point_intervals[..., -1:]), dim=-1)
            lower = torch.cat((point_intervals[..., :1], mids), dim=-1)

            # Stratified samples in those intervals.
            t_rand = torch.rand(
                point_intervals.shape,
                dtype=point_intervals.dtype,
                device=point_intervals.device,
            )
            point_intervals = lower + (upper - lower) * t_rand

        point_intervals = torch.reshape(point_intervals, (ray_count, -1))  # reshape for sanity
        return point_intervals


class HierarchicalPdfSampler(torch.nn.Module):
    """Module that perform Hierarchical sampling (section 5.2)
    Args:
        num_fine_samples (int): Number of depth samples per ray for the fine network.
    """
    def __init__(self, num_fine_samples):
        super(HierarchicalPdfSampler, self).__init__()
        self.num_fine_samples = num_fine_samples
        uniform_x = torch.linspace(0.0, 1.0, steps=self.num_fine_samples)
        uniform_x.requires_grad = False
        self.register_buffer("uniform_x", uniform_x)

    def forward(self, depth_values_coarse, weights, perturb: bool = True):
        """ 
            Inputs:
                depth_values_coarse: (ray_count, num_coarse_samples]) : Depth values of each sampled point along the ray
                weights: (ray_count, num_coarse_samples]) Weights assigned to each sampled color of sampled point along the ray
                perturb_sample: (bool) if True, perform stratified sampling, otherwise perform unifrom sampling. 
            Outputs:
                depth_values_fine: (ray_count, num_coarse_samples + num_fine_samples) : depths of the hierarchical sampled points along the ray
        """
        points_on_rays_mid = 0.5 * (depth_values_coarse[..., 1:] + depth_values_coarse[..., :-1])
        interval_samples = self.sample_pdf(points_on_rays_mid, weights[..., 1:-1], self.uniform_x,
                                           det=(perturb == False)).detach()

        depth_values_fine, _ = torch.sort(torch.cat((depth_values_coarse, interval_samples), dim=-1), dim=-1)
        return depth_values_fine

    def sample_pdf(self, bins, weights, uniform_x, det: bool = False):
        """Hierarchical sampling (section 5.2) for fine sampling
             implementation by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
            Inputs:
                bins: [ray_count, num_coarse_samples - 1] : points_on_rays_mid
                weights: [ray_count, num_coarse_samples - 2] Weights assigned to each sampled color exclude first and last one
                uniform_x: A one-dimensional tensor of size 'num_fine_samples' whose values are evenly spaced from 0 to 1, inclusive.
                det: bool. deterministic or not, if True, perform uniform sampling.
            Outputs:
                samples: [ray_count, num_coarse_samples + num_fine_samples] : depths of the hierarchical sampled points along the ray
        """
        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [ray_count, num_coarse_sample-2]
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat(
            [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
        )  # [ray_count, num_coarse_sample-1] = [ray_count, len(bins)]

        # Take uniform samples
        if det:  # cdf.shape[:-1] get cdf shape exclude last one
            uniform_x = uniform_x.expand(list(cdf.shape[:-1]) + [self.num_fine_samples])  # (ray_count, num_fine_samples)
        else:
            uniform_x = torch.rand(
                list(cdf.shape[:-1]) + [self.num_fine_samples],
                dtype=weights.dtype,
                device=weights.device,
            )

        # Invert CDF
        uniform_x = uniform_x.contiguous().detach()
        cdf = cdf.contiguous().detach()

        inds = torch.searchsorted(cdf, uniform_x, right=True)  # (ray_count, num_fine_samples)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack((below, above), dim=-1)  # (ray_count, num_fine_samples, 2)

        matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])  # (ray_count, num_fine_samples, num_coarse_sample-1)
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (uniform_x - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

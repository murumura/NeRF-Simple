import warnings
import os
import torch
from .Network import NeRFMLP
from .Renderer import VolumeRenderer
from .Sampler import RaySampleInterval, HierarchicalPdfSampler, intervals_to_ray_points
from typing import Tuple
import mcubes


class NeRFModel(torch.nn.Module):
    """The full NeRF model"""
    def __init__(
        self,
        perturb: bool = True,
        lindisp: bool = False,
        # for coarse/fine mlp
        NeRF_Depth: int = 8,
        NeRF_Width: int = 256,
        NeRF_in_channels_xyz: int = 3,
        NeRF_in_channels_dir: int = 3,
        skips: list = [4],
        use_encoder: bool = True,
        pos_encoder_name: str = "position_encoder",
        num_freqs_xyz: int = 10,
        log_sampling_xyz: bool = True,
        num_freqs_dir: int = 4,
        log_sampling_dir: bool = True,
        # for volume rendering
        train_radiance_field_noise_std: float = 0.0,
        val_radiance_field_noise_std: float = 0.0,
        white_background: float = False,
        attenuation_threshold: float = 1e-3,
        # for depth sampler, pdf sampler
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
    ):
        """Constructor
        Args:
            perturb:(bool) wether to perturb depth sampling points (random offset)
            lindisp:(bool) sampling linearly in disparity rather than depth, using in LLFF dataset
            NeRF_Depth:(int) number of layers for density (sigma) encoder
            NeRF_Width:(int) number of hidden units in each layer
            NeRF_in_channels_xyz:(int) number of input channels for xyz
            NeRF_in_channels_dir:(int) number of input channels for direction
            skips:(list) add skip connection in the Dth layer
            use_encoder: (bool) using position encoder for coordinate embedding
            pos_encoder_name(string): registered name for postion encoder
            num_freqs_xyz(int): frequency of encoding functions to use in the positional encoding of the coordinates.
            log_sampling_xyz(bool): whether or not to perform log sampling in the positional encoding of the coordinates.
            num_freqs_dir(int): frequency of encoding functions to use in the positional encoding of the directions.
            log_sampling_dir(bool): whether or not to perform log sampling in the positional encoding of the directions.
            train_radiance_field_noise_std: (0 < float < 1): factor to perturb the model prediction of sigma when training.
            val_radiance_field_noise_std:  (0 < float < 1): factor to perturb the model prediction of sigma when evaluating.
            white_background: (bool): 
                if true, By calculating the complement (`1-acc_map`) of the weight accumulation of each light (acc_map), 
                if the complement is larger, the light is more likely to pass through the free-space, 
                add the value of r, g, and b of each ray to make the value close to 1 (white).
            attenuation_threshold: (float) penetrability threshold, transparency of sampled point along the ray transparency,
                (`T_i`) below this threshold will be regarded as not intersecting with the object (sampled point in free space).
            num_coarse_samples (int): Number of depth samples per ray for the coarse network.
            num_fine_samples (int): Number of depth samples per ray for the fine network.
        """
        super(NeRFModel, self).__init__()
        # Device on which to run.
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            warnings.warn(
                "Please note that although executing on CPU is supported," +
                "the training is unlikely to finish in reasonable time."
            )
            self.device = "cpu"

        self.perturb = perturb
        self.lindisp = lindisp
        self.mlp_coarse = NeRFMLP(
            NeRF_Depth=NeRF_Depth,
            NeRF_Width=NeRF_Width,
            NeRF_in_channels_xyz=NeRF_in_channels_xyz,
            NeRF_in_channels_dir=NeRF_in_channels_dir,
            skips=skips,
            use_encoder=use_encoder,
            pos_encoder_name=pos_encoder_name,
            num_freqs_xyz=num_freqs_xyz,
            log_sampling_xyz=log_sampling_xyz,
            num_freqs_dir=num_freqs_dir,
            log_sampling_dir=log_sampling_dir
        )
        self.mlp_fine = NeRFMLP(
            NeRF_Depth=NeRF_Depth,
            NeRF_Width=NeRF_Width,
            NeRF_in_channels_xyz=NeRF_in_channels_xyz,
            NeRF_in_channels_dir=NeRF_in_channels_dir,
            skips=skips,
            use_encoder=use_encoder,
            pos_encoder_name=pos_encoder_name,
            num_freqs_xyz=num_freqs_xyz,
            log_sampling_xyz=log_sampling_xyz,
            num_freqs_dir=num_freqs_dir,
            log_sampling_dir=log_sampling_dir
        )
        self.mlp_coarse.to(self.device)
        self.mlp_fine.to(self.device)
        self.volume_renderer = VolumeRenderer(
            train_radiance_field_noise_std=train_radiance_field_noise_std,
            val_radiance_field_noise_std=val_radiance_field_noise_std,
            white_background=white_background,
            attenuation_threshold=attenuation_threshold,
            device=self.device
        )
        self.depth_sampler = RaySampleInterval(num_samples=num_coarse_samples, device=self.device)
        self.hierarchical_pdf_sampler = HierarchicalPdfSampler(num_fine_samples=num_fine_samples)

    def forward(self, rays: torch.Tensor) -> Tuple[dict, dict]:
        """ Performs the coarse and fine rendering passes of the radiance field 
        Args:
            rays: (num_rays, 8) tensor containing ray origin, ray direcitons, ray near and for
        Returns:
            dict containing the outputs of the rendering results:
                fine_bundle (dict) Rendering result from fine mlp.
                coarse_bundle (dict)  Rendering result from coarse mlp.
        """
        ray_origins, ray_dirs = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, N_sample, 3)
        ray_near, ray_far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, N_sample, 1)
        num_rays = ray_dirs.shape[0]
        coarse_bundle, fine_bundle = None, None
        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse", "fine"):
            if renderer_pass == "coarse":
                # Generating depth samples
                ray_depth_values = self.depth_sampler(
                    ray_count=num_rays, near=ray_near, far=ray_far, lindisp=self.lindisp, perturb=self.perturb
                )  # (num_rays, self.num_coarse_samples)
            elif renderer_pass == "fine":
                ray_depth_values = self.hierarchical_pdf_sampler(
                    depth_values_coarse=ray_depth_values, weights=coarse_bundle['weights'], perturb=self.perturb
                )  # (num_rays, self.num_coarse_samples)

            # Samples points across each ray (num_rays, N_sample, 3)
            ray_points = intervals_to_ray_points(
                point_intervals=ray_depth_values, ray_directions=ray_dirs, ray_origins=ray_origins
            )
            # Expand rays to match batch size
            expanded_ray_dirs = ray_dirs[..., None, :].expand_as(ray_points).contiguous()

            if renderer_pass == "coarse":
                # Coarse inference
                coarse_radiance = self.mlp_coarse(xyz=ray_points, dir=expanded_ray_dirs)
                coarse_bundle = self.volume_renderer(
                    radiance_field=coarse_radiance, depth_values=ray_depth_values, ray_directions=ray_dirs
                )
            elif renderer_pass == "fine":
                # Fine inference
                fine_radiance = self.mlp_fine(xyz=ray_points, dir=expanded_ray_dirs)
                fine_bundle = self.volume_renderer(
                    radiance_field=fine_radiance, depth_values=ray_depth_values, ray_directions=ray_dirs
                )

        return coarse_bundle, fine_bundle

    def forward_sigma(self, points):
        return self.mlp_fine(points, sigma_only=True)

    @torch.no_grad()
    def extract_mesh(
        self, out_dir, mesh_name, iso_level: int = 32, sample_resolution: int = 128, limit: float = 1.2, batch_size: int = 1024
    ):
        """ Output extracted mesh from radiance field 
        Args:
            out_dir(str): Path to store output mesh file.
            iso_level(int): Iso-level value for triangulation
            mesh_name(str): output mesh name(in obj.)
            sample_resolution: (int) Sampling resolution for marching cubes, increase it for higher level of detail.
            limit:(float) limits in -xyz to xyz for marching cubes 3D grid.
        """

        # define helper function for batchify 3D grid
        def batchify_grids(*data, batch_size=1024, device="cpu"):
            assert all(sample is None or sample.shape[0] == data[0].shape[0] for sample in data), \
                "Sizes of tensors must match for dimension 0."
            # Data size and current batch offset
            size, batch_offset = (data[0].shape[0], 0)
            while batch_offset < size:
                # Subsample slice
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                # Yield each subsample, and move to available device
                yield [sample[batch_slice].to(device) if sample is not None else sample for sample in data]
                batch_offset += batch_size

        sample_resolution = (sample_resolution, ) * 3

        # Create sample tiles
        grid_xyz = [torch.linspace(-limit, limit, num) for num in sample_resolution]

        # Generate 3D samples and flatten it
        grids3d_flat = torch.stack(torch.meshgrid(*grid_xyz), -1).view(-1, 3).float()

        sigmas_samples = []

        # Batchify 3D grids
        for (sampled_grids, ) in batchify_grids(grids3d_flat, batch_size=batch_size, device=self.device):
            # Query radiance batch
            sigma_batch = self.forward_sigma(points=sampled_grids)
            # Accumulate radiance
            sigmas_samples.append(sigma_batch.cpu())

        # Output Radiance 3D grid (density)
        sigmas = torch.cat(sigmas_samples, 0).view(*sample_resolution).contiguous().detach().numpy()
        # Density boundaries
        min_a, max_a, std_a = sigmas.min(), sigmas.max(), sigmas.std()

        # Adaptive iso level
        iso_level = min(max(iso_level, min_a + std_a), max_a - std_a)
        print(f"Min density {min_a}, Max density: {max_a}, Mean density {sigmas.mean()}")
        print(f"Querying based on iso level: {iso_level}")

        # Marhcing cubes
        vertices, triangles = mcubes.marching_cubes(sigmas, iso_level)

        # Create mesh output dir
        os.makedirs(os.path.join(out_dir, 'meshes'), exist_ok=True)
        if not mesh_name.endswith(".obj"):
            mesh_name += ".obj"
        mesh_output = os.path.join(out_dir, 'meshes', f'{mesh_name}')

        # Mesh Extraction
        mcubes.export_obj(vertices, triangles, mesh_output)

    def save_state(self):
        """Return network state dict"""
        checkpoint_dict = {
            "mlp_coarse_state_dict": self.mlp_coarse.state_dict(),
            "mlp_fine_state_dict": self.mlp_fine.state_dict(),
        }
        return checkpoint_dict

    def load_state(self, checkpoint_dict):
        """load network state dict"""
        self.mlp_coarse.load_state_dict(checkpoint_dict['mlp_coarse_state_dict'])
        self.mlp_fine.load_state_dict(checkpoint_dict['mlp_fine_state_dict'])

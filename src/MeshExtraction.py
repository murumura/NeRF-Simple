import configargparse
import os
import logging
import numpy as np
import torch
import mcubes
from pytorch3d.structures import Meshes
from utils import boolean_string
from opt import get_options
from utils import (clear_and_create_folder, Logger)
from models.nerf import NeRFSystem
class MeshExtractor():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = 'cpu'
        # Device
        if args.cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mesh_output = os.path.join(args.out_dir, args.exp_name, args.mesh_name)
        self.NeRFSystem = NeRFSystem(args, logger)
        if self.args.N_importance_samples > 0:
            # load fine model only
            module_load = ("model_fine_state_dict",)
            self.NeRFSystem.load_ckpt(
                ckpts=args.ckpt,
                nn_module_name=module_load
            )
        else:
            # load coarse model only
            module_load = ("model_coarse_state_dict",)
            self.NeRFSystem.load_ckpt(
                ckpts=args.ckpt,
                nn_module_name=module_load
            )
        self.NeRFSystem.eval_mode()
        logging.info(f'''Mesh Extraction logging:
            device:                    {self.device}
            mesh output(.obj):         {self.mesh_output}
        ''')

    def extract_radiance(self, sample_resolution):
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
            
        assert (isinstance(sample_resolution, tuple) or isinstance(sample_resolution, list) or isinstance(sample_resolution, int)), \
            "sample_resolution arg should be either iterable or int."
        if isinstance(sample_resolution, int):
            sample_resolution = (sample_resolution,) * 3
        else:
            assert (len(sample_resolution) == 3), "sample_resolution arg should be of length 3, number of axes for 3D"
        # Create sample tiles
        grid_xyz = [torch.linspace(-args.limit, args.limit, num) for num in sample_resolution]
        # Generate 3D samples and flatten it 
        grids3d_flat = torch.stack(torch.meshgrid(*grid_xyz), -1).view(-1, 3).float()
        radiance_samples = []
        # Batchify 3D grids 
        for (sampled_grids,) in batchify_grids(grids3d_flat, batch_size=self.args.batch_size, device=self.device):
            # Query radiance batch
            radiance_batch = self.NeRFSystem.query_points(
                points=sampled_grids, 
                rays=sampled_grids
            )
            # Accumulate radiance
            radiance_samples.append(radiance_batch.cpu())
        
        # Output Radiance 3D grid (rgb + density)
        radiance = torch.cat(radiance_samples, 0).view(*sample_resolution, 4).contiguous().detach().numpy()

        return radiance # (sample_resolution, sample_resolution, sample_resolution, 4)

    def extract_geometry(self):
        # Sample points based on the grid
        radiance = self.extract_radiance(self.args.res)
        # Density grid
        sigma = radiance[..., 3]
        # Adaptive iso level
        iso_value = self.extract_iso_level(sigma)
        # Extracting iso-surface triangulated
        vertices, triangles = mcubes.marching_cubes(sigma, iso_value)
        
        return vertices, triangles

    def extract_iso_level(self, density):
        # Density boundaries
        min_a, max_a, std_a = density.min(), density.max(), density.std()

        # Adaptive iso level
        iso_value = min(max(self.args.iso_level, min_a + std_a), max_a - std_a)
        print(f"Min density {min_a}, Max density: {max_a}, Mean density {density.mean()}")
        print(f"Querying based on iso level: {iso_value}")

        return iso_value

    def export_mesh_via_MC(self):
        # Mesh Extraction
        vertices, triangles = self.extract_geometry()
        mcubes.export_obj(vertices, triangles, self.mesh_output)
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_options().parse_args()
    logger = Logger(
        args=args,
        output_dir=os.path.join(args.out_dir, args.exp_name)
    )
    mesh_extractor = MeshExtractor(args, logger)
    mesh_extractor.export_mesh_via_MC()
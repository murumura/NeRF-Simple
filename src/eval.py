import argparse
import os
import torch
from models.nerf import NeRFModel
from pyhocon import ConfigFactory

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default='./configs/base.conf')
    parser.add_argument('--ckpt_path', type=str, default='./pretrained/lego.pth')
    parser.add_argument('--output_path', type=str, default='./output')
    # mesh extraction options
    parser.add_argument("--mesh_name", type=str, default="mesh.obj", help="Mesh name to be generated.")
    parser.add_argument("--iso_level", type=float, default=32, help="Iso-level value for triangulation")
    parser.add_argument("--limit", type=float, default=1.2, help="Limits in -xyz to xyz for marching cubes 3D grid.")
    parser.add_argument(
        "--sample_resolution",
        type=int,
        default=128,
        help="Sampling resolution for marching cubes, increase it for higher level of detail."
    )
    args = parser.parse_args()

    # Configuration
    conf_path = args.conf_path
    f = open(conf_path)
    conf_text = f.read()
    f.close()
    conf = ConfigFactory.parse_string(conf_text)

    # Model
    nerf_model = NeRFModel(
        perturb=conf.get_bool('model.perturb', default=True),
        lindisp=conf.get_bool('model.lindisp', default=False),
        num_coarse_samples=conf.get_int('model.num_coarse_samples', default=64),
        num_fine_samples=conf.get_int('model.num_fine_samples', default=128),
        **conf.model.nerf,
        **conf.model.volume_renderer
    )
    checkpoint_dict = torch.load(args.ckpt_path, map_location=nerf_model.device)
    nerf_model.load_state(checkpoint_dict=checkpoint_dict)
    # Create mesh output dir
    os.makedirs(args.output_path, exist_ok=True)
    nerf_model.extract_mesh(
        out_dir=args.output_path,
        mesh_name=args.mesh_name,
        iso_level=args.iso_level,
        sample_resolution=args.sample_resolution,
        limit=args.limit
    )

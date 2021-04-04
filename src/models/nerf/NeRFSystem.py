import torch
from .NeRFhelper import intervals_to_ray_points, cast_to_depth_image
from .Sampler import (SamplePDF, RaySampleInterval)
from .Models import NeRF
from .Embedder import Embedder
from .Renderer import VolumeRenderer
from datasets import BlenderDataset, dataset_dict 
class NeRFSystem(torch.nn.Module):
    def __init__(self, args, logger):
        super(NeRFSystem, self).__init__()
        self.args = args
        self.logger = logger
        # Device
        if args.cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Postion encoding module
        self.embedder_xyz = Embedder(
            input_channels=3,
            num_freqs=args.num_encoding_fn_xyz,
            log_scale=args.log_sampling_xyz
        ) 
        self.embedder_dir = Embedder(
            input_channels=3,
            num_freqs=args.num_encoding_fn_dir,
            log_scale=args.log_sampling_dir
        ) 

        # Initialize a coarse-resolution model.
        self.model_coarse = NeRF().to(self.device)
        # If a fine-resolution model is specified, initialize it.
        self.model_fine = None
        if args.N_importance_samples > 0:
            self.model_fine = NeRF().to(self.device)

        # sampler module
        self.sample_pdf = SamplePDF(args.N_importance_samples)
        self.sampler = RaySampleInterval(args.N_coarse_samples, self.device)

        # rendering module
        self.volume_renderer = \
            VolumeRenderer(
                train_radiance_field_noise_std=args.noise_std, 
                val_radiance_field_noise_std=0, 
                white_background=args.white_bkgd, 
                attenuation_threshold=1e-5,
                device=self.device
            )

        # Criterions
        self.loss = torch.nn.MSELoss()

        # Dataset
        self.train_dataset, self.val_dataset = None, None

        # Mode
        self.training = False

    def train_mode(self):
        self.training = True
        self.model_coarse.train()
        if self.model_fine is not None:
            self.model_fine.train()
        
    def eval_mode(self):
        self.model_coarse.eval()
        if self.model_fine is not None:
            self.model_fine.eval()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.args.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

    def prepare_data(self):
        dataset = dataset_dict[self.args.dataset_type]
        kwargs = {'root_dir': self.args.dataset_dir,
                  'img_wh': tuple(self.args.img_wh)
                }
        # get specified dataset
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        params = list(self.model_coarse.parameters())
        if self.args.N_importance_samples > 0:
            params += list(self.model_fine.parameters())
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params=params,
                                 lr=self.args.learning_rate,
                                 betas=(0.9, 0.999))
        else:
            raise NotImplementedError('args.optimizer type [%s] is not found' % self.args.optimizer)
        return optimizer

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

        
    def forward(self, ray_batch):
        """ Does a prediction for a batch of rays.

        Args:
            x: Tensor of camera rays containing position, direction and bounds.

        Returns: Tensor with the calculated pixel value for each ray.
        """
        def inference(model, embedder_xyz, xyz, embedder_dir, expanded_rays_dir, device):
            """
            Helper function that performs model inference.

            Inputs:
                model: NeRF model (coarse or fine)
                embedder_xyz: embedding module for xyz
                xyz: (N_rays, N_samples_, 3) sampled positions
                    N_samples_ is the number of sampled points in each ray;
                                = N_samples for coarse model
                                = N_samples+N_importance for fine model
                embedder_dir: embedding module for direction
                expanded_rays_dir: (N_rays, N_samples, 3) :expanded ray directions
            Outputs:
                if sigma_only:
                    radience_field:(N_rays, N_samples, 1) 
                else:
                    radience_field:(N_rays, N_samples, 3+1) 
            """
            assert xyz.shape[0]==expanded_rays_dir.shape[0], 'Number of ray differ in coordinate and directions!'
            N_rays = xyz.shape[0]
            assert xyz.shape[1]==expanded_rays_dir.shape[1], 'Number of samples differ in coordinate and directions!'
            N_samples = xyz.shape[1]
            # Flatten the first two dimensions -> (N_rays * N_samples, 3)
            xyz_ = xyz.view(-1, 3) # (N_rays*N_samples_, 3)
            expanded_rays_dir = expanded_rays_dir.contiguous()
            expanded_rays_dir_ =  expanded_rays_dir.view(-1, 3) # (N_rays*N_samples_, 3)

            xyz_embedded = embedder_xyz(xyz_) # (N_rays*N_samples_, 63)
            dir_embedded = embedder_dir(expanded_rays_dir_) # (N_rays*N_samples_, 27)

            xyzdir_embedded = torch.cat((xyz_embedded, dir_embedded), dim=-1).to(device)
            radiance_field = model(xyzdir_embedded)
            radiance_field = torch.reshape(radiance_field, (N_rays,N_samples,-1)).to(device)
            return radiance_field

        ray_origins, ray_directions = ray_batch[:, 0:3], ray_batch[:, 3:6] # both (N_rays, N_sample, 3)
        near, far = ray_batch[:, 6:7], ray_batch[:, 7:8] # both (N_rays, N_sample, 1)
        # is training mode or inference mode?
        train_mode = self.args.train

        # Generating depth samples 
        ray_count = ray_directions.shape[0]
        ray_depth_values = self.sampler(
            ray_count=ray_count,
            near=near,
            far=far,
            lindisp=self.args.lindisp,
            perturb=self.args.perturb
        )
        # Samples across each ray (N_rays, N_sample, 3)
        ray_points = intervals_to_ray_points(ray_depth_values, ray_directions, ray_origins)

        # Expand rays to match batch size
        expanded_ray_directions = ray_directions[..., None, :].expand_as(ray_points)
        # Coarse inference
        coarse_radiance_field = inference(
            model=self.model_coarse,
            embedder_xyz=self.embedder_xyz,
            xyz=ray_points,
            embedder_dir=self.embedder_dir,
            expanded_rays_dir=expanded_ray_directions,
            device=self.device
        )
        coarse_bundle = self.volume_renderer(coarse_radiance_field, ray_depth_values, ray_directions)

        fine_bundle = None
        # Fine inference depth_values_coarse, weights, perturb:
        if self.model_fine is not None:
            fine_ray_depth_values = self.sample_pdf(
                depth_values_coarse=ray_depth_values, 
                weights=coarse_bundle['weights'], 
                perturb=self.args.perturb
            )
            ray_points = intervals_to_ray_points(
                fine_ray_depth_values, 
                ray_directions, 
                ray_origins
            )
            # Expand rays to match batch_size
            expanded_ray_directions = ray_directions[..., None, :].expand_as(ray_points)
            # Fine inference
            # Coarse inference
            fine_radiance_field = inference(
                model=self.model_fine,
                embedder_xyz=self.embedder_xyz,
                xyz=ray_points,
                embedder_dir=self.embedder_dir,
                expanded_rays_dir=expanded_ray_directions,
                device=self.device
            )
            fine_bundle = self.volume_renderer(fine_radiance_field, fine_ray_depth_values, ray_directions)
        
        return coarse_bundle, fine_bundle

    def training_step(self, batch, batch_idx):
        self.train_mode()
        # Unpacking bundle
        rays, rgbs = self.decode_batch(batch)
        rays, rgbs = rays.to(self.device), rgbs.to(self.device)

        # Manual batching, since images are expensive to be kept on GPU
        batch_count = self.args.batch_size

        coarse_loss, fine_loss = 0, 0
        # Forward pass
        coarse_bundle, fine_bundle = self.forward(rays)
        coarse_loss += self.loss(coarse_bundle['rgb_map'], rgbs)
        if self.model_fine is not None:
            fine_loss += self.loss(fine_bundle['rgb_map'], rgbs)
        #  Compute loss
        coarse_loss /= batch_count
        log_vals = {
            "train/coarse_loss": coarse_loss
        }

        loss = coarse_loss
        if self.model_fine is not None:
            #  Compute loss
            fine_loss /= batch_count
            loss += fine_loss
            log_vals = {
                **log_vals,
                "train/fine_loss": fine_loss
            }

        return {
            "loss": loss,
            "log": {
                "train/loss": loss,
                **log_vals
            }
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Do batched inference on rays using chunk."""
        self.eval_mode()
        rays, rgbs = self.decode_batch(batch)
        rays, rgbs = rays.to(self.device), rgbs.to(self.device)
        rays = rays.squeeze() # (H*W, 8)
        rgbs = rgbs.squeeze() # (H*W, 3)
        HxW = rgbs.shape[0]
        coarse_bundle, fine_bundle = {}, {}
        # Forward pass render ray in chunk to avoid OOM
        for i in range(0, HxW, self.args.chunk_size):
            coarse_bundle_chunk, fine_bundle_chunk = \
                self.forward(rays[i:i+self.args.chunk_size])
            for k, v in coarse_bundle_chunk.items():
                if k not in coarse_bundle:
                    coarse_bundle[k] = []
                coarse_bundle[k].append(v)
            
            if fine_bundle_chunk is not None:
                for k, v in fine_bundle_chunk.items():
                    if k not in fine_bundle:
                        fine_bundle[k] = []
                    fine_bundle[k].append(v)

        # flatten return rendering bundle
        coarse_bundle = {k : torch.cat(coarse_bundle[k], 0) for k in coarse_bundle}
        if bool(fine_bundle):
            fine_bundle = {k : torch.cat(fine_bundle[k], 0) for k in fine_bundle}

        bundle = fine_bundle if bool(fine_bundle) else coarse_bundle
        log = {'val_loss': self.loss(bundle['rgb_map'], rgbs)}
        # logging
        self.logger.add(
            category='val_loss', 
            k='rendering', 
            v=log['val_loss'], 
            it=batch_idx
        )
        render_img = (batch_idx % self.args.render_img_every == 0 and batch_idx > 0)
        if render_img:
            W, H = self.args.img_wh
            img = bundle['rgb_map'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = cast_to_depth_image(bundle[f'depth_map'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.add_imgs(stack, 'val_GT_pred_depth', batch_idx)

        render_video = (batch_idx % self.args.render_video_every == 0 and batch_idx > 0)
        if render_video:
            pass
        if self.training:
            self.train_mode()
        return log

    @torch.no_grad()
    def validation_epoch_end(self, epoch_number):
        """Do logging and saving checkpoints."""
        saving_ckpt = (epoch_number % self.args.save_every == 0 and epoch_number > 0)
        if saving_ckpt:
            checkpoint_dict = {
                "epoch": epoch_number,
                "model_coarse_state_dict": self.model_coarse.state_dict(),
                "model_fine_state_dict": None if not self.model_fine else self.model_fine.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }
            self.logger.save_states(checkpoint_dict, epoch_number)

        fine_model_loss_last = \
            self.logger.get_last(category='val_loss', k='rendering')
        mean_rendering_loss = \
            self.logger.get_mean(category='val_loss', k='rendering')
        return {
            "saving_ckpt": saving_ckpt,
            "last_rendering_loss": fine_model_loss_last,
            "mean_rendering_loss": mean_rendering_loss
        }
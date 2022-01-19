import os
import torch
import datetime
import logging
from .NeRFhelper import (cast_to_depth_image, cast_to_disparity_image, psnr, save_img)
from .NeRF import NeRFModel
from datasets import dataset_dict
from pyhocon import ConfigFactory
from torch.utils.tensorboard import SummaryWriter


class NeRFSystem(torch.nn.Module):
    def __init__(self, args):
        """Constructor
        Args:
            args: parsing arguments
        """
        super(NeRFSystem, self).__init__()

        # https://stackoverflow.com/questions/56967754/how-to-store-and-retrieve-a-dictionary-of-tuples-in-config-parser-python
        def parse_int_tuple(input):
            return tuple(int(k.strip()) for k in input[1:-1].split(','))

        # Configuration
        self.conf_path = args.conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', args.exp_tag)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))

        # Misc.
        self.save_every = self.conf.get_int('train.save_every', default=10000)
        self.validate_every = self.conf.get_int('train.validate_every', default=5000)
        self.chunk_size = self.conf.get_int('train.chunk_size', default=1024)
        self.batch_size = self.conf.get_int('train.batch_size', default=2048)
        self.img_wh = parse_int_tuple(self.conf.get_string('dataset.img_wh', default="(400, 400)"))
        self.end_iter = self.conf.get_int('train.end_iter', default=150000)
        # Criterions
        self.loss = torch.nn.MSELoss()
        self.psnr = psnr

        # Dataset
        self.train_dataset, self.val_dataset = None, None
        self.dataset_type = self.conf.dataset.dataset_type
        self.dataset_dir = self.conf.dataset.dataset_dir
        # Model
        self.model = NeRFModel(
            perturb=self.conf.get_bool('model.perturb', default=True),
            lindisp=self.conf.get_bool('model.lindisp', default=False),
            num_coarse_samples=self.conf.get_int('model.num_coarse_samples', default=64),
            num_fine_samples=self.conf.get_int('model.num_fine_samples', default=128),
            **self.conf.model.nerf,
            **self.conf.model.volume_renderer
        )
        self.device = self.model.device
        # Mode
        self.training = True

        # Optimizer
        self.optimizer = None

        self.iter_n = 0

    def train_mode(self):
        self.training = True
        self.model.mlp_coarse.train()
        self.model.mlp_fine.train()

    def eval_mode(self):
        self.model.mlp_coarse.eval()
        self.model.mlp_fine.eval()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True, num_workers=4, batch_size=self.batch_size, pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True
        )

    def prepare_data(self):
        dataset = dataset_dict[self.dataset_type]
        kwargs = {'root_dir': self.dataset_dir, 'img_wh': tuple(self.img_wh)}
        # get specified dataset
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def get_scheduler(self, optimizer):
        # Following the original code, we use exponential decay of the
        # learning rate: current_lr = base_lr * gamma ** (epoch / step_size)
        gamma = self.conf.get_float('train.scheduler.options.gamma', default=0.1)
        step_size = self.conf.get_int('train.scheduler.options.step_size', default=450000)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: gamma**(step / step_size))

    def configure_optimizers(self):
        params = []
        params += list(self.model.mlp_coarse.parameters())
        params += list(self.model.mlp_fine.parameters())
        lr = self.conf.get_float('train.optimizer.lr', 5e-4)
        self.optimizer = getattr(torch.optim, self.conf.train.optimizer.type)\
            (params, lr=lr)

        if hasattr(torch.optim.lr_scheduler, self.conf.train.scheduler.type):
            scheduler = getattr(torch.optim.lr_scheduler, self.conf.train.scheduler.type)\
                (self.optimizer, **self.conf.train.scheduler.options)
        else:
            scheduler = self.get_scheduler(self.optimizer)

        return self.optimizer, scheduler

    def decode_batch(self, batch):
        rays = batch['rays']  # (B, 8)
        rgbs = batch['rgbs']  # (B, 3)
        return rays, rgbs

    def training_step(self, batch, iter_n):
        self.train_mode()
        # Unpacking bundle
        rays, rgbs = self.decode_batch(batch)
        rays, rgbs = rays.to(self.device), rgbs.to(self.device)
        coarse_loss, fine_loss = 0, 0

        # Forward pass
        coarse_bundle, fine_bundle = self.model(rays)

        coarse_loss += self.loss(coarse_bundle['rgb_map'], rgbs)
        coarse_psnr = self.psnr(coarse_loss.item())

        fine_loss += self.loss(fine_bundle['rgb_map'], rgbs)
        fine_psnr = self.psnr(fine_loss.item())

        # logging loss/psnr for coarse network
        log_vals = {
            "train/coarse_loss": coarse_loss,
            "train/coarse_psnr": coarse_psnr,
            "train/fine_loss": fine_loss,
            "train/fine_psnr": fine_psnr
        }
        loss = coarse_loss + fine_loss
        self.writer.add_scalar('Loss/loss', loss, iter_n)
        self.writer.add_scalar('Loss/fine_mse', fine_loss, iter_n)
        self.writer.add_scalar('Loss/coarse_mse', coarse_loss, iter_n)
        self.writer.add_scalar('Statistics/fine_psnr', fine_psnr, iter_n)
        self.writer.add_scalar('Statistics/coarse_psnr', coarse_psnr, iter_n)

        return {"loss": loss, "log": {"train/loss": loss, **log_vals, "train/lr": self.optimizer.param_groups[0]['lr']}}

    @torch.no_grad()
    def validation_step(self, batch, iter_n):
        """Do batched inference on rays using chunk."""
        self.eval_mode()
        rays, rgbs = self.decode_batch(batch)
        rays, rgbs = rays.to(self.device), rgbs.to(self.device)
        rays = rays.squeeze()  # (H*W, 8)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        HxW = rgbs.shape[0]
        coarse_bundle, fine_bundle = {}, {}
        # Forward pass render ray in chunk to avoid OOM
        for i in range(0, HxW, self.chunk_size):
            coarse_bundle_chunk, fine_bundle_chunk = \
                self.model(rays[i:i+self.chunk_size])
            for k, v in coarse_bundle_chunk.items():
                if k not in coarse_bundle:
                    coarse_bundle[k] = []
                coarse_bundle[k].append(v)

            for k, v in fine_bundle_chunk.items():
                if k not in fine_bundle:
                    fine_bundle[k] = []
                fine_bundle[k].append(v)

        # flatten return rendering bundle
        coarse_bundle = {k: torch.cat(coarse_bundle[k], 0) for k in coarse_bundle}
        fine_bundle = {k: torch.cat(fine_bundle[k], 0) for k in fine_bundle}

        val_loss = self.loss(fine_bundle['rgb_map'], rgbs)
        val_psnr = self.psnr(val_loss.item())

        logging.info(f'val-loss:{val_loss}, val-psnr:{val_psnr}.')

        os.makedirs(os.path.join(self.base_exp_dir, 'validations'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'disparity'), exist_ok=True)

        W, H = self.img_wh
        img = fine_bundle['rgb_map'].view(H, W, 3).cpu()
        img = img.permute(2, 0, 1)  # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        depth = cast_to_depth_image(fine_bundle[f'depth_map'].view(H, W))  # (3, H, W)
        stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
        disp = cast_to_disparity_image(fine_bundle[f'disp_map'].view(H, W))  # (3, H, W)

        save_img(stack, os.path.join(os.path.join(self.base_exp_dir, 'validations'), f'validations%08d.png' % iter_n))
        save_img(disp, os.path.join(os.path.join(self.base_exp_dir, 'disparity'), f'disparity%08d.png' % iter_n))

        saving_ckpt = (iter_n % self.save_every == 0 and iter_n > 0)
        if saving_ckpt:
            self.save_checkpoint(iter_n)

        self.train_mode()

    def save_checkpoint(self, iter_n):
        mlp_states = self.model.save_state()
        checkpoint = {
            **mlp_states,
            'optimizer': self.optimizer.state_dict(),
            'iter_n': iter_n,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(iter_n)))

    def load_ckpt(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.model.load_state(checkpoint)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_n = checkpoint['iter_n']
        logging.info('End')

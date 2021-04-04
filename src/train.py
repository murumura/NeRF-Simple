import glob
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm
import logging
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from opt import get_options
from utils import (clear_and_create_folder, Logger)
from models.nerf import NeRFSystem
class Trainer(): 
    def __init__(self, batch_size, epochs, validate_freq):
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.validate_freq = validate_freq
        if args.cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'''Training logging:
            device:                    {self.device}
            batch_size:                {self.batch_size}
            num_epochs:                {self.num_epochs}
        ''')
    def fit(self, NeRFSystem):

        def data_loop(dl):
            """
            Loop an iterable infinitely
            """
            while True:
                for x in iter(dl):
                    yield x
        ##### prepare training/validation datasets ##### 
        NeRFSystem.prepare_data()
        train_dataloader = NeRFSystem.train_dataloader()
        validation_dataloader = NeRFSystem.val_dataloader()
        validation_iterator = data_loop(validation_dataloader)
        ##### configure optimizer #####
        optimizer = NeRFSystem.configure_optimizers()
        ##### setup #####
        total_time = time.time()
        n_rays = len(train_dataloader) # rays
        n_val_imgs = len(validation_dataloader) # images
        #####  Core optimization loop  #####
        for epoch in range(self.num_epochs):
            batch_time = time.time()
            with tqdm(total=n_rays, desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='rays') as pbar:
                for batch_idx, batch in enumerate(train_dataloader):
                    loss_dict = NeRFSystem.training_step(batch,batch_idx)
                    loss = loss_dict['loss']
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    ###   update learning rate  (@todo) ###
                    pbar.set_postfix(
                        **{
                        'Train loss(batch)'     : loss_dict['log']['train/loss'].item(),
                        'Coarse loss (batch)'   : loss_dict['log']['train/coarse_loss'].item(),
                        'Fine loss (batch)'     : loss_dict['log']['train/fine_loss'].item()
                        }
                    )
                    pbar.update()
                    # Validation
                    if (epoch * n_rays + batch_idx) % (self.validate_freq) == 0 and batch_idx > 0:
                        tqdm.write("[VAL] =======> Iter: " + str(batch_idx))
                        val_batch = next(validation_iterator)
                        val_loss_dict = NeRFSystem.validation_step(val_batch, epoch * n_rays + batch_idx)
                        tqdm.write(f"[VAL-LOSS] =======> {val_loss_dict['val_loss']}")
            ### Rest is logging ###
            epoch_summary = NeRFSystem.validation_epoch_end(epoch_number=epoch)
            if epoch_summary.get("saving_ckpt", False):
                tqdm.write("================== Saved Checkpoint =================")
            tqdm.write(f"[FINAL-RENDERING-LOSS] =======> {epoch_summary['last_rendering_loss']}")
            tqdm.write(f"[MEAN-RENDERING-LOSS] =======> {epoch_summary['mean_rendering_loss']}")
            delta_time_batch = time.time() - batch_time
            tqdm.write(f"[VAL-LOSS] =======> {val_loss_dict['val_loss']}")
            tqdm.write(f"================== End of Epoch {epoch}, Duration : {delta_time_batch} =================")
        #####  We are done!  #####
        logging.info(f'Done! Total time spent:{time.time() - total_time}')
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_options().parse_args()
    logger = Logger(
        log_dir=os.path.join(args.log_dir, args.exp_name),
        output_dir=os.path.join(args.out_dir, args.exp_name),
        state_dir=os.path.join(args.state_dir, args.exp_name),
        args=args
    )
    system = NeRFSystem(args, logger)
    nerf_trainer = Trainer(
        batch_size=args.batch_size, 
        epochs=args.epochs,
        validate_freq=args.validate_every
    )
    nerf_trainer.fit(system)
    
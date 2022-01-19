import torch
import time
import logging
import argparse
import datetime
from tqdm import tqdm
from models.nerf import NeRFSystem


class NeRFTrainer():
    def __init__(self):
        logging.info(f'Start training {datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')

    def parse_log_dict(self, log_dict) -> dict:
        return {
            name:val.item() if callable((getattr(val, "item", None))) \
                else val for name, val in log_dict["log"].items()
        }

    def fit(self, NeRFSystem):
        def data_loop(dl):
            """
            Loop an iterable infinitely
            """
            while True:
                for x in iter(dl):
                    yield x

        ##### set up training misc. ######
        self.batch_size = NeRFSystem.batch_size
        self.end_iter = NeRFSystem.end_iter
        self.validate_freq = NeRFSystem.validate_every
        self.iter_n = NeRFSystem.iter_n
        ##### prepare training/validation datasets #####
        NeRFSystem.prepare_data()
        train_dataloader = NeRFSystem.train_dataloader()
        validation_dataloader = NeRFSystem.val_dataloader()
        validation_iterator = data_loop(validation_dataloader)
        ##### configure optimizer #####
        optimizer, scheduler = NeRFSystem.configure_optimizers()
        ##### setup #####
        total_time = time.time()
        n_rays = len(train_dataloader)  # rays
        n_val_imgs = len(validation_dataloader)  # images
        #####  Core optimization loop  #####
        while self.iter_n < self.end_iter:
            batch_time = time.time()
            with tqdm(total=n_rays, desc=f'iters {self.iter_n + 1}/{self.end_iter}', unit='rays') as pbar:
                for batch_idx, batch in enumerate(train_dataloader):
                    self.iter_n += 1
                    loss_dict = NeRFSystem.training_step(batch, self.iter_n)
                    loss = loss_dict['loss']
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ###   update learning rate  ###
                    scheduler.step()
                    pbar.set_postfix(**self.parse_log_dict(loss_dict))
                    pbar.update()
                    # Validation
                    if (self.iter_n) % (self.validate_freq) == 0 and batch_idx > 0:
                        tqdm.write("[VAL] =======> Iter: " + str(self.iter_n))
                        val_batch = next(validation_iterator)
                        NeRFSystem.validation_step(val_batch, self.iter_n)

            delta_time_batch = time.time() - batch_time
            tqdm.write(f"================== End of Training {self.iter_n}, Duration : {delta_time_batch} =================")

        #####  We are done!  #####
        logging.info(f'Done! Total time spent:{time.time() - total_time}')


if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default='./configs/base.conf')
    parser.add_argument('--exp_tag', type=str, default='exp_tag')
    parser.add_argument('--is_continue', default=False, action="store_true")

    args = parser.parse_args()

    system = NeRFSystem(args)
    nerf_trainer = NeRFTrainer()
    nerf_trainer.fit(system)

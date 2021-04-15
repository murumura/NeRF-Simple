import logging
import os
import torch
import torchvision
from .misc import clear_and_create_folder

class Logger(object):
    def __init__(
        self, 
        args,
        output_dir,
        log_dir=None,
        state_dir=None
    ):
        self.stats = dict()
        # setup folder for video/image/gif/mesh output
        self.output_dir = output_dir
        clear_and_create_folder(self.output_dir)
        if args.train:
            self.log_dir = log_dir
            clear_and_create_folder(self.log_dir)
            log_args_copy = os.path.join(self.log_dir, 'args.txt')
            with open(log_args_copy, 'w') as file:
                for arg in sorted(vars(args)):
                    attr = getattr(args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
                logging.info(f'Copy configuration arguments to {log_args_copy}')

            if args.config is not None:
                log_config_copy = os.path.join(self.log_dir, 'config.txt')
                with open(log_config_copy, 'w') as file:
                    file.write(open(args.config, 'r').read())
                logging.info(f'Copy configuration file to {log_config_copy}')

            # setup folder for state saving
            self.state_dir = state_dir
            clear_and_create_folder(self.state_dir)
            logging.info(f''':
                Training state will be saved to:       {self.state_dir}
                Training output will be saved to:      {self.output_dir}
            ''')
        

    def add(self, category, k, v, it):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        self.stats[category][k].append((it, v))

    def get(self, category, k, it, default=0.):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][it][1]
    
    def get_last(self, category, k):
        return self.get(category,k,-1)

    def get_mean(self, category, k, default=0.):
        if category not in self.stats:
            return default

        return torch.stack([self.get(category, k, i) for i in range(len(self.stats[category][k]))]).mean()

    def add_imgs(self, imgs, class_name, it):
        outfile = os.path.join(self.output_dir, f'{class_name}%08d.png' % it)
        imgs = imgs / 2 + 0.5
        imgs = torchvision.utils.make_grid(imgs)
        torchvision.utils.save_image(imgs.clone(), outfile, nrow=8)
        
    def save_states(self, ckpt_dict, e_it):
        torch.save(
                ckpt_dict,
                os.path.join(self.state_dir, "checkpoint" + str(e_it).zfill(5) + ".ckpt"),
        )

    def load_ckpt(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            logging.WARNING(f''':
                Warning: file  {ckpt_path} does not exist!     
            ''')
            return
        try:
            with open(ckpt_path, 'rb') as f:
                self.stats = torch.load(f)
        except EOFError:
            logging.WARNING(f''':
                Warning: log file corrupted!     
            ''')
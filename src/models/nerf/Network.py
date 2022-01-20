import torch
from .Encoder import get_encoder


class NeRFMLP(torch.nn.Module):
    def __init__(
        self,
        NeRF_Depth=8,
        NeRF_Width=256,
        NeRF_in_channels_xyz=3,
        NeRF_in_channels_dir=3,
        skips=[4],
        use_encoder=True,
        **kwargs
    ):
        """
        NeRF_Depth: number of layers for density (sigma) encoder
        NeRF_Width: number of hidden units in each layer
        NeRF_in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        NeRF_in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        use_encoder: (bool) using position encoder for coordinate embedding
        skips: add skip connection in the Dth layer
        """
        super(NeRFMLP, self).__init__()
        self.D = NeRF_Depth
        self.W = NeRF_Width
        self.in_channels_xyz = NeRF_in_channels_xyz
        self.in_channels_dir = NeRF_in_channels_dir
        self.skips = skips
        self.xyz_encoder = None
        self.dir_encoder = None
        if use_encoder:
            encoder_name = kwargs.get('encoder_name', "position_encoder")
            # point coordinate position encoder
            num_freqs_xyz = kwargs.get("num_freqs_xyz", 10)
            log_sampling_xyz = kwargs.get("log_sampling_xyz", True)
            self.xyz_encoder, self.in_channels_xyz = get_encoder(
                encoder_name=encoder_name,
                input_channels=NeRF_in_channels_xyz,
                num_freqs=num_freqs_xyz,
                log_scale=log_sampling_xyz
            )
            # direction position encoder
            num_freqs_dir = kwargs.get("num_freqs_dir", 4)
            log_sampling_dir = kwargs.get("log_sampling_dir", True)
            self.dir_encoder, self.in_channels_dir = get_encoder(
                encoder_name=encoder_name,
                input_channels=NeRF_in_channels_dir,
                num_freqs=num_freqs_dir,
                log_scale=log_sampling_dir
            )
        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = torch.nn.Linear(self.in_channels_xyz, self.W)
            elif i in skips:
                layer = torch.nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = torch.nn.Linear(self.W, self.W)
            layer = torch.nn.Sequential(layer, torch.nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = torch.nn.Linear(self.W, self.W)

        # direction encoding layers
        self.dir_encoding = torch.nn.Sequential(torch.nn.Linear(self.W + self.in_channels_dir, self.W // 2), torch.nn.ReLU(True))

        # output layers
        self.sigma = torch.nn.Linear(self.W, 1)
        self.rgb = torch.nn.Sequential(torch.nn.Linear(self.W // 2, 3), torch.nn.Sigmoid())

    def forward(self, xyz, dir=None, sigma_only=False):
        """
        Encodes input (xyz + dir) to rgb + sigma 
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)
        Outputs:
            if sigma_only:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if self.xyz_encoder is not None:
            xyz = self.xyz_encoder(xyz)
        if self.dir_encoder is not None and dir is not None:
            dir = self.dir_encoder(dir)

        input_xyz = xyz

        for i in range(self.D):
            if i in self.skips:
                xyz = torch.cat([input_xyz, xyz], -1)
            xyz = getattr(self, f"xyz_encoding_{i+1}")(xyz)

        sigma = self.sigma(xyz)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz)

        dir_encoding_input = torch.cat([xyz_encoding_final, dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)
        return out

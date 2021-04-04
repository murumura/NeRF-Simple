import torch

class Embedder(torch.nn.Module):
    """Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...) (Different from the paper, prepend input 'x' by default)
    Args:
        input_channels (int): number of input channels (3 for both xyz and direction)
        num_freqs (int): `L_d=4` for viewing direcion, `L_x=10` for 3D-coordinate (accroding to Positional encoding (section 5.1) of NeRF paper)  
        log_scale (bool): First take power of 2 for 0 to 9 and then split equally (log_scale=False) 
                    or choose to generate 0-9 first and then take power of 2 separately (log_scale=True) 
    """
    def __init__(self, input_channels, num_freqs, log_scale=True):
        super(Embedder, self).__init__()
        self.num_freqs = num_freqs
        self.input_channels = input_channels
        self.encode_fn = [torch.sin, torch.cos]
        self.output_channels = input_channels * (len(self.encode_fn) * num_freqs + 1)
        if log_scale:
            self.freq_bands = 2**torch.linspace(0, num_freqs-1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(num_freqs-1), num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (ray_cnt, num_sample, self.in_channels)
        Outputs:
            out: (ray_cnt, num_sample, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.encode_fn:
                out += [func(freq*x)]

        return torch.cat(out, -1)
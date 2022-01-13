import torch
import torch.nn as nn
import torchvision
from Block import Block


class Decoder(nn.Module):
    def __init__(self, chs=((1024, 512), (1024, 512), (1024, 512), (1024, 256), (512, 128), (256, 64), (128, 1))):
        super().__init__()
        self.chs = chs
        self.dec_blocks = nn.ModuleList([Block(chs[i][0], chs[i][1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)):
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            if i != len(self.chs) - 1:
                x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

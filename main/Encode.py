import torch.nn as nn

from Block import Block


class Encoder(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 512, 512, 512, 512)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for i, block in enumerate(self.enc_blocks):
            if i == 0:
                x = block(x, encoder=True, decoder=False, first=True)
            elif i == len(self.enc_blocks):
                ftrs.append(x)
                x = block(x, encoder=True, decoder=False, last=True)
            else:
                ftrs.append(x)
                x = block(x, encoder=True, decoder=False)
        return ftrs

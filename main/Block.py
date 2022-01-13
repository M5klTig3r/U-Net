import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_ch, out_ch,  kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.leaky = nn.LeakyReLU(0.2)
        self.batch = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, encoder=False, decoder=True, first=False, last=False):
        if encoder:
            if first:
                output = self.leaky(self.conv1(x))
            elif last:
                output = self.batch(self.relu(self.conv1(x)))
            else:
                output = self.batch(self.leaky(self.conv1(x)))
        if decoder:
            if last:
                output = self.tanh(self.deconv1(x))
            else:
                output = self.batch(self.relu(self.deconv1(x)))
        return output

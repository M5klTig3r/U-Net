import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from Encode import Encoder
from Decoder import Decoder


class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64, 1), num_class=1,
                 retain_dim=False):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.head = nn.Sequential(nn.ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                  nn.Tanh())
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][0:])
        out = self.head(out)
        return out


def sample_image(n_row, batches_done, current_epoch, real_images):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    gen_images = unet(real_images)
    # TODO - create folder conditionally
    save_image(gen_images.data, f"images/{current_epoch}_{batches_done}.png", nrow=n_row, normalize=True)


dataloader = torch.utils.data.DataLoader(datasets.ImageFolder(
    # "../../iOCT/bigVol_9mm",
    "../../iOCT/bigVol_9mm",
    transform=transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
         transforms.Resize((512, 512)),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])]
    ),
),
    batch_size=1,
    shuffle=True,
)

cuda = True if torch.cuda.is_available() else False
torch.cuda.empty_cache()
unet = UNet()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    print(f"Cuda: {cuda}")
    unet.cuda()

# input image
for epoch in range(300):
    for i, (images, labels) in enumerate(dataloader):

        if i == 0:
            continue

        x = Variable(images.type(FloatTensor))
        output = unet(x)

        print(
            f"[Epoch %d/%d] [Batch %d/%d] [Image size {output.shape}]"
            % (epoch, 300, i, len(dataloader))
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % 100 == 0:
            sample_image(n_row=10, batches_done=batches_done, current_epoch=epoch, real_images=x)

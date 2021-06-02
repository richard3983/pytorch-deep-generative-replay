from torch import nn
from torch.nn import functional as F


class Critic(nn.Module):
    def __init__(self, image_size, image_channel_size, channel_size):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        # self.conv1 = nn.Conv2d(
        #     image_channel_size, channel_size,
        #     kernel_size=4, stride=2, padding=1
        # )
        # self.conv2 = nn.Conv2d(
        #     channel_size, channel_size*2,
        #     kernel_size=4, stride=2, padding=1
        # )
        # self.conv3 = nn.Conv2d(
        #     channel_size*2, channel_size*4,
        #     kernel_size=4, stride=2, padding=1
        # )
        # self.conv4 = nn.Conv2d(
        #     channel_size*4, channel_size*8,
        #     kernel_size=4, stride=1, padding=1,
        # )
        # self.fc = nn.Linear((image_size//8)**2 * channel_size*4*9//8, 1)
        # self.out = nn.Sigmoid()
        
        # self.n_features = (3, 32, 32)
        # nc, ndf = 3, 64

        self.input_layer = nn.Sequential(
            nn.Conv2d(image_channel_size, channel_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(channel_size, channel_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(channel_size * 2, channel_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(channel_size * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_size, image_size, image_channel_size, channel_size):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        # self.fc = nn.Linear(z_size, (image_size//8)**2 * channel_size*8)
        # self.bn0 = nn.BatchNorm2d(channel_size*8)
        # self.bn1 = nn.BatchNorm2d(channel_size*4)
        # self.deconv1 = nn.ConvTranspose2d(
        #     channel_size*8, channel_size*4,
        #     kernel_size=4, stride=2, padding=1
        # )
        # self.bn2 = nn.BatchNorm2d(channel_size*2)
        # self.deconv2 = nn.ConvTranspose2d(
        #     channel_size*4, channel_size*2,
        #     kernel_size=4, stride=2, padding=1,
        # )
        # self.bn3 = nn.BatchNorm2d(channel_size)
        # self.deconv3 = nn.ConvTranspose2d(
        #     channel_size*2, channel_size,
        #     kernel_size=4, stride=2, padding=1
        # )
        # self.deconv4 = nn.ConvTranspose2d(
        #     channel_size, image_channel_size,
        #     kernel_size=3, stride=1, padding=1
        # )
        # self.out = nn.Sigmoid()
        
        # self.n_features = 100
        # self.n_out = (3, 32, 32)
        # nc, nz, ngf = 3, 100, 64

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(z_size, channel_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(channel_size * 8),
            nn.ReLU(True),
        )

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(channel_size * 8, channel_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel_size * 4),
            nn.ReLU(True),
        )

        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d( channel_size * 4, channel_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel_size * 2),
            nn.ReLU(True),
        )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(channel_size * 2, image_channel_size, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(self.image_size, self.z_size, 1, 1)
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

from torch import nn

class TomatoNet(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)
            #m = m.to(dtype=torch.float16)

    def __init__(self, in_channels, n_classes):
        super(TomatoNet, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 16, (in_channels, 7, 7), stride=(1, 1, 1), padding=(int(in_channels / 2), 3, 3))
        self.conv_1x1 = nn.Conv3d(
            1, 16, (in_channels, 1, 1), stride=(1, 1, 1), padding=(int(in_channels / 2), 0, 0))

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv3d(32, 32, (in_channels, 5, 5), padding=(int(in_channels / 2)-1, 2, 2))
        self.conv2 = nn.Conv3d(32, 32, (in_channels, 5, 5), padding=(int(in_channels / 2)-1, 2, 2))
        self.conv3 = nn.Conv3d(32, 32, (in_channels, 5, 5), padding=(int(in_channels / 2), 2, 2))

        # Residual block 2
        self.conv4 = nn.Conv3d(32, 64, (40, 1, 1), padding=0)
        self.conv5 = nn.Conv2d(64, 32, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(32, 32, (1, 1))
        self.conv7 = nn.Conv2d(32, 16, (1, 1))
        self.conv8 = nn.Conv2d(16, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(64)
        self.lrn2 = nn.LocalResponseNorm(64)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))
        #print(x.shape)

        # First residual block
        x_res = F.relu(self.conv2(x))
        #print(x_res.shape)
        x_res = self.conv3(x_res)
        #print(x_res.shape)
        x = F.relu(x_res)
        #print(x.shape)

        
        # Second residual block
        x = F.relu(self.conv4(x))
        #print(x_res.shape)
        x = torch.squeeze(x)
        #print(x_res.shape)
        x = self.conv5(x)
        x = F.relu(x)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        return x
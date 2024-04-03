import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoding3D(nn.Module):
    def __init__(self, D, K, encoding):
        super(Encoding3D, self).__init__()
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.reset_params()
        self.encoding = encoding

        self.fc = nn.Sequential(nn.Linear(D, D), nn.Sigmoid())

    def reset_params(self):
        std1 = 1.0 / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        assert X.size(1) == self.D
        B, D, T, H, W = X.size()
        N = T * H * W
        K = self.K

        # Flatten X to (B, N, D) for processing
        I = X.view(B, D, N).transpose(1, 2).contiguous()

        # Calculate assignment weights A (B, N, K, D)
        A = F.softmax(
            self.scale.view(1, 1, K, D)
            * (I.unsqueeze(2) - self.codewords.view(1, K, D)).pow(2),
            dim=2,
        )

        if not self.encoding:  # Embedding
            E = (A * (I.unsqueeze(2) - self.codewords.view(1, K, D))).sum(1)
            E = E.mean(dim=1)
            gamma = self.fc(E)

            E = (A * (I.unsqueeze(2) - self.codewords.view(1, K, D))).sum(2)
            E = E.transpose(1, 2).contiguous().view(B, D, T, H, W)
            y = gamma.view(B, D, 1, 1, 1)
            E = F.relu_(E + E * y)
        else:  # Encoding
            E = (A * (I.unsqueeze(2) - self.codewords.view(1, K, D))).sum(1)

        return E


class EmbeddingModule3D(nn.Module):
    def __init__(self, in_channels, ncodes=24):
        super(EmbeddingModule3D, self).__init__()
        self.encoding = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            Encoding3D(D=in_channels, K=ncodes, encoding=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(2 * in_channels, in_channels, 1, bias=True),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        encoded = self.encoding(x)
        output = self.conv(torch.cat((x, encoded), dim=1))
        return output


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels),
            )
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class UNet3DWithKEM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_n_filter=64, ncodes=24):
        super(UNet3DWithKEM, self).__init__()

        self.enc1 = ResBlock3D(in_channels, base_n_filter)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ResBlock3D(base_n_filter, base_n_filter * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ResBlock3D(base_n_filter * 2, base_n_filter * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck with KEM
        self.bottleneck = ResBlock3D(base_n_filter * 4, base_n_filter * 8)
        self.kem = EmbeddingModule3D(base_n_filter * 8, ncodes=ncodes)

        self.up3 = nn.ConvTranspose3d(
            base_n_filter * 8, base_n_filter * 4, kernel_size=2, stride=2
        )
        self.dec3 = ResBlock3D(base_n_filter * 8, base_n_filter * 4)

        self.up2 = nn.ConvTranspose3d(
            base_n_filter * 4, base_n_filter * 2, kernel_size=2, stride=2
        )
        self.dec2 = ResBlock3D(base_n_filter * 4, base_n_filter * 2)

        self.up1 = nn.ConvTranspose3d(
            base_n_filter * 2, base_n_filter, kernel_size=2, stride=2
        )
        self.dec1 = ResBlock3D(base_n_filter * 2, base_n_filter)

        self.final_conv = nn.Conv3d(base_n_filter, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        bottleneck = self.bottleneck(pool3)
        kem_output = self.kem(bottleneck)

        up3 = self.up3(kem_output)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        output = self.final_conv(dec1)
        return output

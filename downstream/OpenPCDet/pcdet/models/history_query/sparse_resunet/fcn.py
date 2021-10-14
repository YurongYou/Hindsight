import torch.nn as nn
import MinkowskiEngine as ME

class FullyConvNet2(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, config, D=3):
        super(FullyConvNet2, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=5,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiMaxPooling(
                kernel_size=3,
                dimension=D)
            )
            
        # self.conv2 = nn.Sequential(
        #    ME.MinkowskiConvolution(
        #        in_channels=64,
        #        out_channels=64,
        #        kernel_size=5,
        #        stride=1,
        #        dimension=D),
        #    ME.MinkowskiBatchNorm(out_feat),
        #    ME.MinkowskiReLU(),
        #    ME.MinkowskiMaxPooling(
        #        kernel_size=3,
        #        dimension=D)
        #    )
        
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=out_feat,
            kernel_size=5,
            stride=1,
            dimension=D
        )
        
        # self.linear = ME.MinkowskiLinear(128, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.conv2(out)
        out = self.conv3(out)
        return out

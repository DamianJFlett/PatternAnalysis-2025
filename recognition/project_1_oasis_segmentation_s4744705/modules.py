import torch
import torch.nn as nn
import torch.nn.functional as F



class ImprovedUnet(nn.Module):
    """
    Architecture as per [1] converted to 2D as best as possible is(all Convs 3*3):
    conv -> (context module) -> stride 2 conv -> (context module) -> stride 2 conv -> 
    (context module) -> stride 2 conv -> (context module) stride 2 conv -> (context module)
    #TODO: Make sure that all architectures are described completely. separating into classes makes it easy to describe blocks at a time
    """
    def __init__(self, dropout_prob = 0.3):
        super().__init__()
        self.dropout_prob = 0.3

class ContextBlock(nn.Module):
    """
    Context Module (pre-activation residual block) as per [1] converted to 2d:
    Batch norm -> activation function (relu) -> 3*3 conv
    -> dropout
    -> Batch norm -> activation function (relu) -> 3*3 conv -> add on residual
    """
    def __init__(self, in_channels, out_channels, dropout_prob):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1) # 3* 3 convulution

        self.dropout = nn.Dropout2d(dropout_prob)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = self.residual(x)
        
        output = F.relu(self.bn1(x))
        output = self.conv1(output)

        output = self.dropout(output)

        output = F.relu(self.bn2(output))
        output = self.conv2(output)

        return output + residual


class LocalisationBlock(nn.Module):
    """
    Localisation Module as per [1] converted to 2d:
    3*3 conv, then 1*1 conv, then of course activation function and batch norm
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))))

def dice_loss(predicted, target):
    pass
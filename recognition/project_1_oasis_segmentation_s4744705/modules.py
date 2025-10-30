import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedUnet(nn.Module):
    """
    Architecture as per [1] converted to 2D as best as possible is(all Convs 3*3):
    conv -> (context module) -> stride 2 conv -> (context module) -> stride 2 conv -> 
    (context module) -> stride 2 conv -> (context module) stride 2 conv -> (context module)
    -> (upsample block) -> (upsample block) -> (upsample block)
    where upsample blocks also have the corresponding level of context module output concatenated to their input
    #NOTE: Although the paper uses a softmax in the final layer, it is unnecessary here as we do binary classification
    """
    def __init__(self, dropout_prob = 0.3):
        super().__init__()
        self.dropout_prob = dropout_prob

        self.init_conv = nn.Conv2d(1, 16, 3, padding = 1)
        self.init_cont = ContextBlock(16, 16, dropout_prob)
        self.between0 = nn.Conv2d(16, 32, 3, stride = 2, padding = 1)

        self.cont1 = ContextBlock(32, 32, dropout_prob)
        self.between1 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1)

        self.cont2 = ContextBlock(64, 64, dropout_prob)
        self.between2 = nn.Conv2d(64, 128, 3, stride = 2, padding = 1)

        self.cont3 = ContextBlock(128, 128, dropout_prob)
        self.between3 = nn.Conv2d(128, 256, 3, stride = 2, padding = 1)

        self.cont4 = ContextBlock(256, 256, dropout_prob)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upsample_conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.upsample_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample_conv3 = nn.Conv2d(64, 32, 3, padding=1)

        self.loc1 = LocalisationBlock(256, 128)  # since we are concatenating cont3 to input
        self.loc2 = LocalisationBlock(128, 64) # concatenating con2 to input
        self.loc3 = LocalisationBlock(64, 32) # concatenating cont1 to input

        #segmentation layers, as per paper
        self.seg1 = nn.Conv2d(128, 1, 1)
        self.seg2 = nn.Conv2d(64, 1, 1)
        self.seg3 = nn.Conv2d(32, 1, 1)


    def forward(self, x):
        # Encode 
        x = self.init_conv(x)
        c1 = self.init_cont(x)
        
        x = self.between0(c1)
        c2 = self.cont1(x)
        
        x = self.between1(c2)
        c3 = self.cont2(x)

        x = self.between2(c3)
        c4 = self.cont3(x)
        
        x = self.between3(c4)
        x = self.cont4(x)   
        
        # Decode
        x = self.upsample(x)
        x = self.upsample_conv1(x)
        x = torch.cat([x, c4], dim=1)
        x = self.loc1(x)
        seg1 = self.seg1(x)
        
        x = self.upsample(x)
        x = self.upsample_conv2(x)
        x = torch.cat([x, c3], dim=1)
        x = self.loc2(x)
        seg2 = self.seg2(x)
        
        x = self.upsample(x)
        x = self.upsample_conv3(x)
        x = torch.cat([x, c2], dim=1)
        x = self.loc3(x)
        seg3 = self.seg3(x)
        seg1_up = F.interpolate(seg1, size=256, mode='bilinear', align_corners=True)
        seg2_up = F.interpolate(seg2, size=256, mode='bilinear', align_corners=True)
        seg3_up = F.interpolate(seg3, size=256, mode='bilinear', align_corners=True)
        
        # Element-wise addition of deep supervision outputs
        output = seg1_up + seg2_up + seg3_up
        
        return output

class ContextBlock(nn.Module):
    """
    Context Module (pre-activation residual block) as per [1] converted to 2d:
    instance norm -> activation function (relu) -> 3*3 conv
    -> dropout
    -> instance norm -> activation function (relu) -> 3*3 conv -> add on residual
    where relus are leaky w/ slope 10^-2
    """
    def __init__(self, in_channels, out_channels, dropout_prob):
        """
        Initialise a Context Block.
        in_channels: number of channels in input to block
        out_channels: number of channels in output from block
        dropout_prob: probability of dropping out a node in context blocks
        """
        super().__init__()
        self.instnorm1 = nn.InstanceNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1) # 3* 3 convulution

        self.dropout = nn.Dropout2d(dropout_prob)

        self.instnorm2 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = self.residual(x)
        
        output = F.leaky_relu(self.instnorm1(x), negative_slope=10**(-2))
        output = self.conv1(output)

        output = self.dropout(output)

        output = F.leaky_relu(self.instnorm2(output), negative_slope = 10**(-2))
        output = self.conv2(output)

        return output + residual


class LocalisationBlock(nn.Module):
    """
    Localisation Module as per [1] converted to 2d:
    3*3 conv, then 1*1 conv, then of course activation function and instance norm
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialise a Localisation block
        in_channels: the number of input channels to the block
        out_channels: the number of output channels from the block
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        self.instnorm = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
    def forward(self, x):
        """
        Apply in sequence the operations of a localisation block
        x: input to apply transforms on
        """
        return F.leaky_relu(self.instnorm(self.conv2(self.conv1(x))), negative_slope = 10**(-2))
    
"""
This code taken and modified from lecture code
"""
class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.

    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Sigmoid output from model [B, H, W] (values between 0-1)
            targets: Binary ground truth [B, H, W] (values 0 or 1)
        """
        # Flatten tensors using reshape to handle non-contiguous memory layout
        predictions = torch.sigmoid(predictions)
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1).float()

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - dice_coeff

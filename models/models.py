import torch
import torch.nn as nn
import torchvision
import numpy as np
from . import resnet, resnext, mobilenet
from lib.nn import SynchronizedBatchNorm2d
from . import convcrf as cc
import math

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()
    
    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 1).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)

        acc = acc_sum.float() / (pixel_sum.float() + 1e-10) #When you +falsePos, acc == Jaccard.
        
        #class 1
        v1 = (label == 1).long()
        pred1 = (preds == 1).long()
        anb1 = torch.sum(v1 * pred1)
        try:
            j1 = anb1.float() / (torch.sum(v1).float() + torch.sum(pred1).float() - anb1.float() + 1e-10)
        except:
            j1 = 0
        #class 2 
        v2 = (label == 2).long()
        pred2 = (preds == 2).long()
        anb2 = torch.sum(v2 * pred2)
        try: 
            j2 = anb2.float() / (torch.sum(v2).float() + torch.sum(pred2).float() - anb2.float() + 1e-10)
        except:
            j2 = 0

        if j1 > 1:
            j1 = 0

        if j2 > 1:
            j2 = 0
        
        #class 3
        v3 = (label == 3).long()
        pred3 = (preds == 3).long()
        anb3 = torch.sum(v3 * pred3)
        try:
            j3 = anb3.float() / (torch.sum(v3).float() + torch.sum(pred3).float() - anb3.float() + 1e-10)
        except:
            j3 = 0

        j1 = j1 if j1 <= 1 else 0
        j2 = j2 if j2 <= 1 else 0
        j3 = j3 if j3 <= 1 else 0
        
        jaccard = [j1, j2, j3] 
        
        return acc, jaccard

    def jaccard(self, pred, label): #ACCURACY THAT TAKES INTO ACCOUNT BOTH TP AND FP.
        AnB = torch.sum(pred.long() & label) #TAKE THE AND
        return AnB/(pred.view(-1).sum().float() + label.view(-1).sum().float() - AnB)
    
class SegmentationModule(SegmentationModuleBase):
    def __init__(self, unet, crit):
        super(SegmentationModule, self).__init__()
        self.crit = crit
        self.unet = unet
     
    def forward(self, feed_dict, *, mode='train'):
        assert mode in ['train', 'val', 'test']
            
        #training
        if segSize == 'train':
            pred = self.unet(feed_dict['image'])
            loss = self.crit(pred, feed_dict['mask'].long(), epoch=epoch)
            acc = self.pixel_acc(torch.round(nn.functional.softmax(pred, dim=1)).long(), feed_dict['mask'].long())
            return loss, acc
            
        #test
        elif segSize == 'test':
            p = self.unet(feed_dict['image'])
            pred = nn.functional.softmax(p, dim=1)
            return pred

        #validation
        else:
            p = self.unet(feed_dict['image'])
            loss = self.crit(p, feed_dict['mask'].long().unsqueeze(0))
            pred = nn.functional.softmax(p, dim=1)
            return pred, loss

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    
    def build_unet(self, num_class=1, arch='unet', weights=''):
        arch = arch.lower()

        if arch == 'unet':
            unet = VanillaNet(num_classes=num_class)
        elif arch == 'denseunet':
            unet = DenseUNet(num_classes=num_class)
        elif arch == 'residualunet':
            unet = ResidualUNet(num_classes=num_class)
        else:
            raise Exception('Architecture undefined!')
        
        if len(weights) > 0:
            unet.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
            print("Loaded pretrained UNet weights.")
        print('Loaded weights for unet')
        return unet

class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]



### ---- UNets ---- ###

class CenterBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(CenterBlock, self).__init__()
        self.in_channels = in_channels
        self.is_deconv=is_deconv

        if self.is_deconv: #Dense Block
            self.conv1 = conv3x3_bn_relu(in_channels, middle_channels)
            self.conv2 = conv3x3_bn_relu(in_channels+middle_channels, middle_channels)
            self.convUp = nn.Sequential(
                                nn.ConvTranspose2d(in_channels+2*middle_channels, out_channels,
                                                kernel_size=4, stride=2, padding=1),
                                nn.ReLU(inplace=True)
            )
        
        else:
            self.convUp = nn.Unsample(scale_factor=2, mode='bilinear', align_corners=True),
            self.conv1 = conv3x3_bn_relu(in_channels, middle_channels),
            self.conv2 = conv3x3_bn_relu(in_channels+middle_channels, out_channels)  
            
    def forward(self, x):
        tmp = []
        if self.is_deconv == False:
            convUp = self.convUp(x); tmp.append(convUp)
            conv1 = self.conv1(convUp); tmp.append(conv1)
            conv2 = self.conv2(torch.cat(tmp, 1))
            return conv2

        else: 
            tmp.append(x)
            conv1 = self.conv1(x); tmp.append(conv1)
            conv2 = self.conv2(torch.cat(tmp, 1)); tmp.append(conv2)
            convUp = self.convUp(torch.cat(tmp, 1))
            return convUp

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        
        if is_deconv:
            self.block = nn.Sequential(
                conv3x3_bn_relu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv3x3_bn_relu(in_channels, middle_channels),
                conv3x3_bn_relu(middle_channels, out_channels),
            )

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        return self.block(x)
    
class SkipConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        
        ## initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        return self.block(x)

#ResNet + UNet
class ResidualUNet(nn.Module):
    def __init__(self, num_classes=2, num_filters=32, pretrained=True, is_deconv=True):
        super(ResidualUNet, self).__init__()
        
        self.num_classes = num_classes
        
        self.pool = nn.MaxPool2d(2,2)
               
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        
        #The layers below aren't assigned for resnet34, print out all layers and assign accordingly. Sorry!
        self.conv1 = nn.Sequential(
                      self.encoder.layer0,
                      self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = nn.Sequential(
                        self.encoder.layer4,
                        self.encoder.layer5,
                        self.encoder.layer6,
                        self.encoder.layer7,
                        self.encoder.layer8
        )

        
        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(32 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(16 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = conv3x3_bn_relu(num_filters, num_filters)
        self.final = nn.Sequential(nn.BatchNorm2d(num_filters),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num_filters, self.num_classes, kernel_size=1)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))
        
        dec5 = self.dec5(torch.cat([center, self.pad(conv5, center)], 1)) #dim=1 because NCHW, channels is dim 1.
        dec4 = self.dec4(torch.cat([dec5, self.pad(conv4, dec5)], 1))
        dec3 = self.dec3(torch.cat([dec4, self.pad(conv3, dec4)], 1))
        dec2 = self.dec2(torch.cat([dec3, self.pad(conv2, dec3)], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.pad(self.dec0(dec1), x)

        x_out = self.final(dec0)
        return x_out

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                     diffY // 2, diffY - diffY //2))

class DenseUNet(nn.Module): #densenet
    def __init__(self, num_classes=2, num_filters=32, pretrained=True, is_deconv=True):
        super(DenseUNet, self).__init__()

        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2,2)
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        self.avgpool4 = nn.AvgPool2d(4, stride=4)
        self.avgpool8 = nn.AvgPool2d(8, stride=8)
        self.encoder = torchvision.models.densenet121(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = nn.Sequential(self.encoder.features.denseblock1,
                                   self.encoder.features.transition1)
        self.conv3 = nn.Sequential(self.encoder.features.denseblock2,
                                   self.encoder.features.transition2)
        self.conv4 = nn.Sequential(self.encoder.features.denseblock3,
                                   self.encoder.features.transition3)
        self.conv5 = nn.Sequential(self.encoder.features.denseblock4,
                                   self.encoder.features.norm5)

        self.center = DecoderBlock(1024, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = conv3x3_bn_relu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, 
                                    self.pad(conv5, center)],1))
        dec4 = self.dec4(torch.cat([dec5, 
                                    self.pad(conv4, dec5)],1))
        dec3 = self.dec3(torch.cat([dec4, 
                                    self.pad(conv3, dec4)],1))
        dec2 = self.dec2(torch.cat([dec3, 
                                    self.pad(conv2, dec3)], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.pad(self.dec0(dec1), x)
        
        x_out = self.final(dec0)
        return x_out

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY //2))

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                        diffY // 2, diffY - diffY //2))


class VanillaUNet(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super(VanillaUNet, self).__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        x_out = self.final(dec1)

        return x_out

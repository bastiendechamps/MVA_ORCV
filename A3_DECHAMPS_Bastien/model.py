import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

nclasses = 20 

class BaseResnet101(nn.Module):
    def __init__(self, n_classes=20, h=1024):
        super(BaseResnet101, self).__init__()
        self.resnet = models.resnet101(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        n_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_features, h)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(h, n_classes)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.resnet(x))))


class Resnext50(nn.Module):
    def __init__(self, n_classes=20, h=1024, last_conv=False):
        super(Resnext50, self).__init__()
        self.resnext = models.resnext50_32x4d(pretrained=True)

        for child, layer in self.resnext.named_children():
            if child != 'layer4':
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = last_conv
        
        n_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(n_features, h)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(h, n_classes)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.resnext(x))))


class Resnext101WSL(nn.Module):
    def __init__(self, n_classes=20, h=1024, last_conv=False):
        super(Resnext101WSL, self).__init__()
        self.resnext = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')

        for child, layer in self.resnext.named_children():
            if child != 'layer4':
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = last_conv
        
        n_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(n_features, h)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(h, n_classes)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.resnext(x))))


class InceptionV3(nn.Module):
    def __init__(self, n_classes=20):
        super(InceptionV3, self).__init__()
        self.incept = models.inception_v3()
        state_dict = torch.load('iNat_2018_InceptionV3.pth.tar')['state_dict']
        self.incept.load_state_dict(state_dict)
        self.incept.aux_logits = False
        
        for param in self.incept.parameters():
            param.requires_grad = False

        num_ftrs = self.incept.fc.in_features
        self.incept.fc = nn.Linear(num_ftrs, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.incept(x)
        return self.fc2(self.dropout(F.relu(x)))



# Taken from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
    
    elif model_name == "resnext101":
        """ ResNext101
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

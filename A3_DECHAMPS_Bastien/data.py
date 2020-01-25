import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

resnet_train_transforms = transforms.Compose([
    transforms.Resize(size=224),
    transforms.CenterCrop(size=224),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
])

resnet_test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
])

inceptionv3_train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
])

inceptionv3_test_transforms = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
])

inceptionv3_params = {
    'batch_size': 32,
    'momentum': 0.9,
    'lr': 0.045,
    'misc': 'RMSProp(0.9, eps=1.0), exp lr decay 0.94 every 2 epochs, grad clipping for stability',
}
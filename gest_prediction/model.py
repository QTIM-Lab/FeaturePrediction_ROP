import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

VERBOSE = False
SAVE_EVERY = 2

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

def inceptionnet(num_classes):
    model = models.inception_v3(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    num_features = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Sequential(nn.Linear(num_features, 1024),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, num_classes))
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 1024),
                            nn.Dropout(0.5),
                            nn.Linear(1024, num_classes))
    return model

# inception net v1
def googlenet_v1(num_classes):
    model = models.googlenet(pretrained=True)
    num_features = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Sequential(nn.Linear(num_features, 1024),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, num_classes))
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 1024),
                            nn.Dropout(0.5),
                            nn.Linear(1024, num_classes))
    return model

def vgg16(num_classes):
    vgg16 = models.vgg16(pretrained=True)
    for param in vgg16.features.parameters():
        param.require_grad = False
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 128), 
                     nn.Dropout(0.5),
                     nn.Linear(128, num_classes)])
    vgg16.classifier = nn.Sequential(*features)
    return vgg16

class cnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3)

        self.pool = nn.MaxPool2d((2,2))

        # self.fc1 = nn.Linear(64, 500)
        # self.fc2 = nn.Linear(500, 100)
        # self.fc3 = nn.Linear(100, 20)
        # self.fc4 = nn.Linear(20, num_classes)

        self.fc1 = nn.Linear(93312, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)

        if CUDA:
            self = self.cuda()

    def forward(self, x):
        # input x = [B, in_channels (Ci), H, W]
        # print("x", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.size())
        # x = torch.squeeze(x, 2)
        x = x.view(-1, 64)
        print("view", x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        # h = self.conv1(x) 
        # # print("conv1", h.size())
        # h = F.relu(h)
        # h = self.pool(h)
        # # print("pool1", h.size())
        # h = self.conv2(h)
        # # print("conv2", h.size())
        # h = F.relu(h)
        # h = self.pool(h)
        # # print("pool2", h.size())
        # h = self.conv3(h)
        # # print("conv3", h.size())
        # h = F.relu(h)
        # h = self.pool(h)
        # # print("pool3", h.size())
        # h = self.pool(h)
        # # print("pool4", h.size())
        # h = self.pool(h)
        # h = self.pool(h)
        # # print("pool5", h.size())
        # h = self.pool(h)
        # print("pool6", h.size())
        # h = h.squeeze()
        # print("before fc", h.size())
        # h = self.fc1(h)
        # # h = F.relu(h)
        # h = self.dropout1(h)
        # h = self.fc2(h)
        # # h = F.relu(h)
        # h = self.dropout2(h)
        # h = self.fc3(h)
        # # h = F.relu(h)
        # y = self.fc4(h)
        # # print("last fc", h)
        # # y = F.relu(h)
        # print("out", y)
        # print("y", y.size())
        # return y

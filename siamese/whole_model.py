import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchsummary import summary


def ConvBlock3(inp, out, stride, pad):
    """
    3x3 ConvNet building block with different activations support.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=3, stride=stride, padding=pad),
        nn.BatchNorm2d(out, eps=1e-3),
        nn.ReLU(True)
    )


def weights_init_uniform(m):
    """
    Initializes the weights using kaiming method.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)


class Branch(nn.Module):
    def __init__(self, bw):
        super().__init__()
        self.block1 = nn.Sequential(ConvBlock3(1, bw, 2, 0),
                                    ConvBlock3(bw, bw, 1, 0),
                                    ConvBlock3(bw, bw, 1, 0),
                                    nn.MaxPool2d(2)
                                    )

        self.block2 = nn.Sequential(ConvBlock3(bw, bw * 2, 1, 0),
                                    ConvBlock3(bw * 2, bw * 2, 1, 0),
                                    nn.MaxPool2d(2)
                                    )

        self.block3 = ConvBlock3(bw * 2, bw * 4, 1, 0)

    def forward(self, x):
        o1 = self.block1(x)
        o2 = self.block2(o1)
        o3 = self.block3(o2)
        return F.avg_pool2d(o3, o3.size()[2:]).view(x.size(0), -1)


# transfer learning
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(1024, 1, True)

    def forward(self, x):
        return self.Linear(x)


class OsteoWholeModel(nn.Module):
    def __init__(self, bw, drop, num_classes=1, use_w_init=True, use_clinic=False, use_acm=False, return_fc=False):
        super().__init__()
        self.branch = Branch(bw, use_acm=use_acm)
        self.use_clinic = use_clinic
        self.return_fc = return_fc
        if self.use_clinic:
            print("Use clinic data !")

        softmaxer = nn.Sigmoid() if num_classes == 2 else nn.Softmax()

        fc_in = 2 * bw * 4 + 4 if self.use_clinic else 2 * bw * 4

        if drop > 0:
            self.final = nn.Sequential(
                nn.Dropout(p=drop),
                nn.Linear(fc_in, num_classes),
                softmaxer
            )

        else:
            self.final = nn.Sequential(
                nn.Linear(fc_in, num_classes),
                softmaxer
            )

        # Custom weights initialization
        if use_w_init:
            self.apply(weights_init_uniform)

    def forward(self, x1, x2=None, clinic=None, save=False):
        # Shared weights
        o1 = self.branch(x1)  # left
        o2 = self.branch(x2) if x2 is not None else o1

        if save:
            self.left_value = o1
            self.right_value = o2

        feats = torch.cat([o1, o2, clinic], 1) if self.use_clinic else torch.cat([o1, o2], 1)

        if self.return_fc:
            return self.final(feats), feats
        else:
            return self.final(feats)


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = (3, 512, 512)

    model.classifier = BinaryClassifier()

    # print("After editing classifier", model)

    with torch.no_grad():
        input_image = np.ones((512, 512))
        input_image = np.broadcast_to(input_image, (1, 3, 512, 512))
        input_image = torch.tensor(input_image, dtype=torch.float).to(device)
        output = model(input_image)

        out_tr = (F.sigmoid(output) > torch.Tensor([0.5])).float()

        print(F.sigmoid(output))
        print(output.size(), output)
    # pass

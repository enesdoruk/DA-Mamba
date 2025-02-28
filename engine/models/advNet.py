import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class LocalAdv(nn.Module):
    def __init__(self,in_channels,context=False):
        super(LocalAdv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels//4, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels//4, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          return F.sigmoid(x)


class GlobalAdv(nn.Module):
    def __init__(self,in_channels, context=False):
        super(GlobalAdv, self).__init__()
        self.in_channels = in_channels
        self.conv1 = conv3x3(self.in_channels, self.in_channels, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.conv2 = conv3x3(self.in_channels, self.in_channels//2, stride=2)
        self.bn2 = nn.BatchNorm2d(in_channels//2)
        self.conv3 = conv3x3(self.in_channels//2, self.in_channels//2, stride=2)
        self.bn3 = nn.BatchNorm2d(self.in_channels//2)
        self.fc = nn.Linear(self.in_channels//2,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,self.in_channels//2)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          return x
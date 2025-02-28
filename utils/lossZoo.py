import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import wandb



class EntropyKD(nn.Module):
    def __init__(self, T=1) -> None:
        super(EntropyKD, self).__init__()

        self.T = T
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x_s, x_t, x_ts, x_st, i):
        xs = x_ts.view(x_ts.shape[0], x_ts.shape[1], -1)
        xt = x_st.view(x_st.shape[0], x_st.shape[1], -1)

        xt_margin = self.get_margin_from_BN(xt)

        loss, xts_ent = self.distillation_loss(xs, xt, self.T, 1, xt_margin)
        wandb.log({f"lay_dist/layer_{i}": loss.item()})
        return loss, xts_ent

    def distillation_loss(self, source, target, T, dims,  margin):
        target = torch.max(target/T, margin.view(target.shape[0],1,1))
        entropy = F.softmax(source/T, dim=dims) * F.log_softmax(target/T, dim=dims)
        channel_entropy = -1 * torch.sum(entropy, dim=dims)
        batch_mean_entropy = torch.mean(channel_entropy, dim=1)
        loss = torch.mean(batch_mean_entropy)
        return loss, entropy

    def get_margin_from_BN(self, bn):
        margin = []
        std = bn.flatten(start_dim=1).std(dim=-1)
        mean = bn.flatten(start_dim=1).mean(dim=-1)
        for (ss, mm) in zip(std, mean):
            ss = abs(ss.cpu().detach().numpy())
            mm = mm.cpu().detach().numpy()
            if norm.cdf(-mm / ss) > 0.001:
                margin.append(- ss * math.exp(- (mm / ss) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-mm / ss) + mm)
            else:
                margin.append(-3 * ss)
        return torch.FloatTensor(margin).to(std.device, non_blocking=True)
    


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, domain):
        B, C = input.shape
        probs = F.softmax(input, 1)
        label = F.one_hot(torch.tensor(domain).repeat(B).to('cuda:0'), self.num_classes).float()

        loss = -(probs.log() * label).sum(-1)
        return loss.mean(-1)


class FocalLoss(nn.Module):
    def __init__(self, num_classes=2, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, input, domain):
        B, C = input.shape
        probs = F.softmax(input, 1)
        label = F.one_hot(torch.tensor(domain).repeat(B).to('cuda:0'), self.num_classes).float()

        probs = (probs * label).sum(-1)
        loss = -torch.pow(1 - probs, self.gamma) * probs.log()
        return loss.mean()


def adv_loss(locl_st, locl_ts,
            globl_st, globl_ts,
            CE, FL):
    
    loss =  ( 
				torch.mean(locl_st ** 2) +
				torch.mean(1-locl_ts ** 2) +
				FL(globl_st, domain=0) +
				FL(globl_ts, domain=1) 
			)

    return loss

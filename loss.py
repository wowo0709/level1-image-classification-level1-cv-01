import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=18, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=18, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

'''
0:	 83
1:	 83
2:	 109
3:	 109
4:	 410
5:	 410
6:	 415
7:	 545
8:	 556
9:	 556
10:	 725
11:	 725
12:	 817
13:	 817
14:	 2050
15:	 2780
16:	 3625
17:	 4085
'''
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=[2780, 2050, 430, 3640, 4065, 535, 556, 410, 86, 728, 813, 107, 556, 410, 86, 728, 813, 107], max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class CustomLDAMLoss(nn.Module):
    def __init__(self, cls_num_list=[2780, 2050, 430, 3640, 4065, 535, 556, 410, 86, 728, 813, 107, 556, 410, 86, 728, 813, 107], max_m=0.5, weight=None, s=30):
        super(CustomLDAMLoss, self).__init__() # !!!
        m_list = 1.0 / np.sqrt(cls_num_list) # !!!
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)



class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate


    def cross_entropy_with_weights(logits, target, weights=None):
        def _log_sum_exp(x):
            # See implementation detail in
            # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
            # b is a shift factor. see link.
            # x.size() = [N, C]:
            b, _ = torch.max(x, 1)
            y = b + torch.log(torch.exp(x - b.expand_as(x)).sum(1))
            # y.size() = [N, 1]. Squeeze to [N] and return
            return y.squeeze(1)
        def _class_select(logits, target):
            # in numpy, this would be logits[:, target].
            batch_size, num_classes = logits.size()
            if target.is_cuda:
                device = target.data.get_device()
                one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                                    .long()
                                                    .repeat(batch_size, 1)
                                                    .cuda(device)
                                                    .eq(target.data.repeat(num_classes, 1).t()))
            else:
                one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                                    .long()
                                                    .repeat(batch_size, 1)
                                                    .eq(target.data.repeat(num_classes, 1).t()))
            return logits.masked_select(one_hot_mask)
        
        assert logits.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1
        loss = _log_sum_exp(logits) - _class_select(logits, target)
        if weights is not None:
            # loss.size() = [N]. Assert weights has the same shape
            assert list(loss.size()) == list(weights.size())
            # Weight the loss
            loss = loss * weights
        return loss


    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return self.cross_entropy_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return self.cross_entropy_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return self.cross_entropy_with_weights(input, target, weights)




_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss,
    'ldam': LDAMLoss,
    'custom_ldam': CustomLDAMLoss,
    'weighted_cross_entropy': WeightedCrossEntropyLoss
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion

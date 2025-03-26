import torch
import torch.nn as nn
import torch.nn.functional as F
from compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from dice import MemoryEfficientSoftDiceLoss

_EPS = 1e-10

# def dice_loss(score, target):
#     target = target.float()
#     smooth = 1e-5
#     intersect = torch.sum(score * target)
#     y_sum = torch.sum(target * target)
#     z_sum = torch.sum(score * score)
#     loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#     loss = 1 - loss
#     return loss

def batch_dice_intersection_union_calculation(preds, labels):

    preds_flat = preds
    preds_flat[preds> 0.5] = 1
    preds_flat[preds <= 0.5] = 0
    preds_flat = preds.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)

    intersection = (preds_flat*labels_flat).sum().item()
    union = (preds_flat.sum().item() + labels_flat.sum().item())
    return intersection, union

def multiclass_dice_coef(input, target, ignore_index = 0):
    # compute multi-class dice coefficient
    total_inter = 0
    total_union = 0
    num_of_labels = target.shape[0]
    for i in range(num_of_labels):
        if i == 0:
            continue

        dice_inter, dice_union = batch_dice_intersection_union_calculation(input[i], target[i])

        total_inter += dice_inter
        total_union += dice_union

    return 2.0 * total_inter / (total_union + 1.0)

def dice_coef(input, target):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        target: a tensor of shape [B, 1, H, W].
        input: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = torch.flatten(input)
    tflat = torch.flatten(target)
    intersection = (iflat * tflat).sum()
    x = iflat.sum()
    y = tflat.sum()

    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def bce_loss(input, target, pos_weight=None):
    """Computes the weighted binary cross-entropy loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, 1, H, W]. Corresponds to
                the raw output or logits of the model.
            pos_weight: a scalar representing the weight attributed
                to the positive class. This is especially useful for
                an imbalanced dataset.
    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(input.float(), target.float(), pos_weight=pos_weight,)
    return bce_loss

def bce_dice_loss(input, target):
    dice = dice_coef(input, target)
    return 0.5*bce_loss(input, target) - dice_coef(input, target)

def my_newPartialCE_loss(pred, target):
    loss = (
        - (torch.log(pred[:, 1, :, :] + _EPS) * target).sum()
        / (target.sum() + _EPS)
    )
    return loss

class my_kl(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        loss = (target * torch.log(target / (output + _EPS) + _EPS)).mean()
        return loss


class HLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = - x * torch.log(x + _EPS)
        b = b.mean()
        return b

class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, weight=True):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):

        input = input.float()
        target = target.float()

        input = torch.squeeze(input, dim=1)
        target = torch.squeeze(target, dim=1)

        shape = input.shape
        N = shape[0]*shape[1]*shape[2]*shape[3]

        sum_p = torch.sum(input)

        if (self.weight == True):
            w = (N - sum_p)/sum_p
        else:
            w = 1

        epsilon = .00001

        crossent = -1.0*torch.sum(w*target*torch.log(input + epsilon) + (1 - target)*torch.log(1 - input + epsilon))

        return crossent/N

class MyLoss(nn.Module):
    def __init__(self, temp=1, n_classes=1):
        super().__init__()
        self.temp = temp
        self.n_classes = n_classes
        self.kl = my_kl()
        if n_classes == 1:
            self.supervised_loss = DC_and_BCE_loss({},
                                                   {'batch_dice': False,
                                                    'do_bg': True, 'smooth': 1e-5},
                                                   use_ignore_label=None,
                                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            self.supervised_loss = DC_and_CE_loss({'batch_dice': False,
                                                   'smooth': 1e-5, 'do_bg': False},
                                                  {}, weight_ce=1, weight_dice=1,
                                                  ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)

        self.entropy = HLoss()

    def forward(
        self, outputA, outputB_F, outputB_W,
        label_F, label_W
    ):
        if self.n_classes == 1:
            predA = F.sigmoid(outputA)
            predB_F = F.sigmoid(outputB_F)
            predB_W = F.sigmoid(outputB_W)

            bce_dice_loss_f = bce_dice_loss(outputA, label_F)
            # Partial CE
            partial_bce_dice_loss_f = bce_dice_loss(outputB_F, label_F)
            partial_bce_dice_loss_w = bce_dice_loss(outputB_W, label_W)

            kl_loss = nn.KLDivLoss()
            dist_loss = kl_loss(F.sigmoid(outputB_F / self.temp), F.sigmoid(outputA / self.temp))
        else:
            predA = F.softmax(outputA, dim=1)
            predB_F = F.softmax(outputB_F, dim=1)
            predB_W = F.softmax(outputB_W, dim=1)

            # strong labels sent to the strong decoder
            sl_sd_loss = self.supervised_loss(outputA, label_F)
            # strong labels sent to the weak decoder
            sl_wd_loss = self.supervised_loss(outputB_F, label_F)
            # strong labels sent to the weak decoder
            wl_wd_loss = self.supervised_loss(outputB_W, label_W)

            kl_loss = nn.KLDivLoss()
            dist_loss = kl_loss(F.log_softmax(outputB_F / self.temp, dim=1),
                                F.softmax(outputA / self.temp, dim=1))

            # dist_loss = self.kl(
            #     F.log_softmax(outputB_F / self.temp, dim=1).exp(),
            #     F.log_softmax(outputA / self.temp, dim=1).exp()
            # )


            # CE of the fully supervised branch
            # ce_loss_f = F.cross_entropy(outputA, label_F)

            # partial_ce_loss_f = my_newPartialCE_loss(predB_F, label_F)
            # partial_ce_loss_f = F.cross_entropy(outputB_F, label_F)
            # partial_ce_loss_w = my_newPartialCE_loss(predB_W, label_W)

            # Distill loss
            # dist_loss = self.kl(
            #     F.log_softmax(outputB_F / self.temp, dim=1).exp(),
            #     F.log_softmax(outputA / self.temp, dim=1).exp()
            # )
        #dist_loss = self.kl(
        #    F.sigmoid(outputB_F / self.temp), F.sigmoid(outputA / self.temp)
        #)

        # entropy
        entropy_loss = self.entropy(
            torch.cat((predB_F, predB_W), dim=0)
        )

        if self.n_classes == 1:
            return (bce_dice_loss_f, partial_bce_dice_loss_f, partial_bce_dice_loss_w,
                    dist_loss, entropy_loss)
        else:
            return (sl_sd_loss, sl_wd_loss, wl_wd_loss,
                    dist_loss, entropy_loss)


class MyLossSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.entropy = HLoss()

    def forward(self, outputF, outputW, labelF, labelW):
        # supervised CE
        ce_loss_f = F.cross_entropy(outputF, labelF)

        predW = F.log_softmax(outputW, dim=1).exp()
        # partial CE
        partial_ce_loss_w = my_newPartialCE_loss(predW, labelW)

        # entropy
        entropy_loss = self.entropy(predW, dim=0)

        return (
            ce_loss_f, partial_ce_loss_w, entropy_loss
        )
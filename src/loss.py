import torch
import torch.nn.functional as F
from torch import Tensor


class LossFunction:
    def calculate(self, source: Tensor, **kwargs):
        pass


# Augmented Cross Entropy Loss
class AugmentedCrossEntropy(LossFunction):
    def __init__(self):
        super(AugmentedCrossEntropy, self).__init__()

    def calculate(self, source: Tensor, target: Tensor, mask: Tensor = None):
        batch_number = source.size(0)
        total_loss = 0
        for i in range(batch_number):
            if mask is not None:
                loss = F.cross_entropy(source[i][mask], target[mask])
            else:
                loss = F.cross_entropy(source[i], target)
            total_loss += loss

        avg_loss = total_loss / batch_number
        return avg_loss

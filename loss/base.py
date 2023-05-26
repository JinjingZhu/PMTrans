import torch 
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
import torch.nn.functional as F

class SoftTargetCrossEntropy(SoftTargetCrossEntropy):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor, target: torch.Tensor, lamda: torch.Tensor=None) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        if lamda is not None:
            loss = loss * lamda / lamda.sum()
        return loss.mean()

class LabelSmoothingCrossEntropy(LabelSmoothingCrossEntropy):
    def __init__(self, smoothing=0.1):
        super().__init__(smoothing)
    
    def forward(self, x: torch.Tensor, target: torch.Tensor, lamda: torch.Tensor=None) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if lamda is not None:
            loss = loss * lamda / lamda.sum()
        return loss.mean()
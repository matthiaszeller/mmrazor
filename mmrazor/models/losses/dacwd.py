
import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .cwd import ChannelWiseDivergence

@MODELS.register_module()
class DimensionAdaptiveChannelWiseDivergence(ChannelWiseDivergence):
    """
    Map teacher to student dimension before calculating CWD loss.
    """

    def __init__(self, tau=1.0, loss_weight=1.0, student_dim=None, teacher_dim=None):
        super().__init__(tau, loss_weight)
        # a simple 1x1 conv to map teacher to student dimension
        assert (student_dim is None) == (teacher_dim is None), 'student_dim and teacher_dim must be both None or not None'
        if student_dim is not None and student_dim != teacher_dim:
            self.projection = nn.Conv2d(teacher_dim, student_dim, 1, 1, 0, bias=False)
        else:
            self.projection = nn.Identity()

        # xavier init to preserve variance
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C_student, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C_teacher, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        # map teacher dimensions
        preds_T = self.projection(preds_T)
        return super().forward(preds_S, preds_T)

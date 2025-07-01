import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoryGradientInducedDistillationLoss(nn.Module):
    def __init__(self, gamma=0.9):
        super(CategoryGradientInducedDistillationLoss, self).__init__()
        self.gamma = gamma

    def forward(self, student_logits, teacher_logits, y_true, task_ids, task_history):
        """
        Args:
            student_logits: [B x C] logits from current model
            teacher_logits: [B x C] logits from previous model
            y_true: [B] true class indices
            task_ids: [B] task index per sample
            task_history: List of past task IDs

        Returns:
            loss: scalar
        """
        device = student_logits.device
        task_mask = torch.tensor([int(tid in task_history) for tid in task_ids]).float().to(device)
        log_p_student = F.log_softmax(student_logits, dim=1)
        p_teacher = F.softmax(teacher_logits, dim=1)

        kl_losses = F.kl_div(log_p_student, p_teacher, reduction='none').sum(dim=1)
        weighted_kl = task_mask * kl_losses
        loss = weighted_kl.mean()
        return loss
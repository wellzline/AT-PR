from attack import Attacker
import torch
import torch.nn as nn
import torch.nn.functional as F

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=2, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce

    def forward(self, logits, targets):
        """
        :param logits: Predictions (model outputs)
        :param targets: Target labels
        :return: Loss value
        """
        onehot_targets = F.one_hot(targets, self.num_classes).float().to(logits.device)
        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1
        )[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, min=0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss

class CW(Attacker):
    def __init__(self, model, config, target=None):
        super(CW, self).__init__(model, config)
        self.target = target
        self.cw_loss = CWLoss(num_classes=config['num_classes'], margin=50, reduce=True)


    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth labels
        :param target: Target labels (for targeted attack)
        :return: Adversarial examples
        """
        x_adv = x.detach().clone()
        if self.config['random_init']:
            x_adv = self._random_init(x_adv)

        for _ in range(self.config['attack_steps']):
            x_adv.requires_grad = True
            self.model.zero_grad()
            logits = self.model(x_adv)  
            loss = self.cw_loss(logits, y)  # torch.Size([256, 100])   torch.Size([256])
            
            loss.backward()
            grad = x_adv.grad.detach()
            grad = grad.sign()

            # Update adversarial example
            x_adv = x_adv + self.config['attack_lr'] * grad

            # Projection to the epsilon-ball
            x_adv = x + torch.clamp(x_adv - x, min=-self.config['eps'], max=self.config['eps'])
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)

        return x_adv


from attack import Attacker
import torch
import torch.nn.functional as F
import random
import torch.nn as nn
import numpy as np
import logging
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class PR(Attacker):
    def __init__(self, model, config, target=None):
        super(PR, self).__init__(model, config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.target = target
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def forward(self, x, y, ae_per_instance=5):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target: Target label (for targeted attacks, not used here)
        :return: Final adversarial example with maximum decision boundary distance
        """
        x_adv_list = []

        pgd = self.generate_pgd_example(x, y, self.config['eps'], self.config['attack_lr'], self.config['attack_steps'])
        pgd_ = pgd.clone()
        x_adv_list.append(pgd)
        while len(x_adv_list) < ae_per_instance:
            epilon = random.uniform(self.config['eps'] - 0.02, self.config['eps'])
            alpha = random.uniform(self.config['attack_lr']  - 0.003, self.config['attack_lr'] + 0.003)
            num_iter = random.randint(self.config['attack_steps'] -5, self.config['attack_steps'] + 7)
            x_adv = self.generate_pgd_example(x, y, epilon, alpha, num_iter)
            x_adv_list.append(x_adv)

        final_pr, pgd_sum, pr_sum = self.pick_best_ae(x, x_adv_list, y)

        matching_items = torch.all(final_pr == pgd_, dim=(1, 2, 3))
        num_matching = matching_items.sum().item()

        different_pr = final_pr[~matching_items]
        different_pgd = pgd_[~matching_items]
        dis2boundary_pr = pr_sum[~matching_items].mean().item()
        dis2boundary_pgd = pgd_sum[~matching_items].mean().item()

        assert different_pr.shape == different_pgd.shape
        if different_pgd.size(0) == 0:
            logging.info(f'Number of identical items in the batch: {num_matching} {pgd_.size(0)} {y[~matching_items].size(0)} {different_pr.shape}')
        else:
            loss_pgd = self.criterion(self.model(different_pgd), y[~matching_items]).detach()  
            loss_pr = self.criterion(self.model(different_pr), y[~matching_items]).detach()  
            diff_distance = torch.norm((different_pr - different_pgd).view(different_pr.size(0), -1), dim=1, p=2)
            logging.info(f'Number of identical items in the batch: {num_matching} {pgd_.size(0)} {different_pr.shape} {y[~matching_items].shape}')
            logging.info('pr_loss: {}'.format(loss_pr))
            logging.info('pgd_loss: {}'.format(loss_pgd))
            logging.info('pr pgd to decision boundary: {}'.format([dis2boundary_pr, dis2boundary_pgd, dis2boundary_pr - dis2boundary_pgd]))
            logging.info('distance of pgd pr: {}'.format(diff_distance.mean(dim = 0)))
            logging.info('*' * 50)
        return final_pr

    def pick_best_ae(self, x, adv_list, y):
        max_distance = torch.zeros(y.size(0)).cuda()
        final_adv_example = adv_list[0] if adv_list else x
        index = 0
        for x_adv in adv_list:
            x_curr = x_adv.clone().detach()
            refine_lr = self.config['attack_lr']
            x_curr = x_curr.requires_grad_(True)
            self.model.zero_grad()
            logits = self.model(x_curr)
            pred = logits.argmax(dim=1)
            is_ae = pred != y
            while is_ae.sum() > y.size(0) * 0.1:
                loss = self.criterion(logits, y)
                self.model.zero_grad()
                loss.backward()

                grad = x_curr.grad.detach()

                x_curr.data[is_ae] = x_curr.data[is_ae] - refine_lr * grad.data[is_ae].sign()
                x_curr.data[is_ae] = torch.clamp(x_curr.data[is_ae], *self.clamp)

                x_curr = x_curr.detach().clone().requires_grad_(True)
                logits = self.model(x_curr)
                pred = logits.argmax(dim=1)
                is_ae = pred != y

            # distance = torch.norm((x_adv - x_curr).view(x_adv.size(0), -1), dim=1, p=float('inf'))
            distance = torch.norm((x_adv - x_curr).view(x_adv.size(0), -1), dim=1, p=2)
            final_adv_example[distance>max_distance] = x_adv[distance>max_distance]
            max_distance[distance>max_distance] = distance[distance>max_distance]
            if index == 0:
                import copy
                pgd_sum = copy.copy(distance)
            index += 1
        return final_adv_example, pgd_sum, max_distance


    def generate_pgd_example(self, x, y, epilon, alpha, num_iter):
        """ Helper function to generate a PGD adversarial example """
        x_adv = x.detach().clone()

        if self.config['random_init']:
            x_adv = self._random_init(x_adv)
        
        for _ in range(num_iter):
            x_adv = x_adv.detach().clone().requires_grad_(True)
            self.model.zero_grad()
            logits = self.model(x_adv)
            loss = self.criterion(logits, y)
            loss.backward()
            grad = x_adv.grad.detach()
            x_adv = x_adv + alpha* grad.sign()
            x_adv = x + torch.clamp(x_adv - x, min=-epilon, max=epilon)
            x_adv = torch.clamp(x_adv, *self.clamp)

        return x_adv
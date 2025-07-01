import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
from autoattack import AutoAttack
import torchattacks
import torch.distributions as dist

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack


def sample_delta(x, epsilon):
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    return x_adv


def corruption_uniform(model, x, y, epsilon=8/255, attack=False):
    x_adv = sample_delta(x, epsilon)
    if attack:
        return x_adv
    else:
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        return loss, logits


def corruption_gaussian(model, x, y, epsilon=8/255, attack=False):
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.clamp(torch.randn_like(x_adv), -epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()
 
    if attack:
        return x_adv
    else:
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        return loss, logits


def corruption_laplace(model, x, y, epsilon=8/255, attack=False):
    # lap = dist.Laplace(loc=torch.tensor(0.0, device=x.device), scale=torch.tensor(epsilon, device=x.device))
    lap = dist.Laplace(loc=torch.tensor(0.0, device=x.device), scale=torch.tensor(1.0, device=x.device))

    x_adv = x.detach().clone()
    x_adv = x_adv + torch.clamp(lap.sample(x_adv.shape).to(x.device), -epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()
 
    if attack:
        return x_adv
    else:
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        return loss, logits


def ERM_DataAug(model, x, y, epsilon=8/255, sample_num = 20):
    loss = 0 
    for _ in range(sample_num):
        x_adv = sample_delta(x, epsilon)
        logits = model(x_adv)
        loss += F.cross_entropy(logits, y, reduction="mean")

    loss = loss / float(sample_num)
    return loss, logits


def fgsm_loss(model, x, y, epsilon=8/255):
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    x_adv.requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y, reduction="mean")
    grad = torch.autograd.grad(loss, [x_adv])[0]
    x_adv = x + epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y, reduction="mean")
    return loss, logits


def pgd_loss(model, x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10, attack=False):
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    for _ in range(attack_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv) 
        loss = F.cross_entropy(logits, y, reduction="mean")
        optimizer.zero_grad()
        loss.backward()                      
        grad = x_adv.grad.detach().sign()
        x_adv = x_adv + step_size * grad
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        
    if attack:
        return x_adv
    else:
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        return loss, logits




def KL_AE(model, x, step_size, epsilon, attack_steps):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.eval()
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
    for _ in range(attack_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                    F.softmax(model(x), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    
    return x_adv


def trades_loss(model, x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10, beta=6.0):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    x_adv = KL_AE(model, x, step_size=step_size, epsilon=epsilon, attack_steps=attack_steps)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    logits = model(x)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                               F.softmax(model(x), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, logits


def mart_loss(model,x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10, beta=5.0):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x)
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
    for _ in range(attack_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    logits = model(x)
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust
    return loss, logits


def CVaR_loss(model, x, y, optimizer, epsilon=8/255, t_step_size=1.0, attack_steps=5, beta=0.5, M=20):
    batch_size = x.shape[0]
    ts = torch.ones(batch_size, device=x.device) 

    def sample_deltas(x, epsilon):
        return 2 * epsilon * torch.rand_like(x) - epsilon

    optimizer.zero_grad()
    for _ in range(attack_steps):

        cvar_loss, indicator_sum = 0, 0
        for _ in range(M):  
            perturbed_x = torch.clamp(x + sample_deltas(x, epsilon), 0, 1) 
            logits = model(perturbed_x)
            curr_loss = F.cross_entropy(logits, y, reduction='none')  
            indicator_sum += torch.where(curr_loss > ts, torch.ones_like(ts), torch.zeros_like(ts))
            cvar_loss += F.relu(curr_loss - ts) 

        indicator_avg = indicator_sum / M
        cvar_loss = (ts + cvar_loss / (M * beta)).mean()

        grad_ts = (1 - (1 / beta) * indicator_avg) / batch_size
        ts = ts - t_step_size * grad_ts 
        
    return cvar_loss, logits




def PR(model, x, y, step_size=2/255, epsilon=8/255, attack_steps=10, max_num = 10):
    x_adv_list = []
    pgd = pgd_loss(model, x, y, step_size=step_size, epsilon=epsilon, attack_steps=attack_steps, attack=True)
    x_adv_list.append(pgd)
    while len(x_adv_list) < max_num:
        epilon = random.uniform(epsilon - 0.02, epsilon)
        alpha = random.uniform(step_size  - 0.003, step_size + 0.003)
        num_iter = random.randint(attack_steps - 2, attack_steps + 10)
        x_adv = pgd_loss(model, x, y, step_size=alpha, epsilon=epilon, attack_steps=num_iter, attack=True)
        x_adv_list.append(x_adv)

    # idx = random.randint(0, len(x_adv_list)-1)
    # final_pr = x_adv_list[idx]
    final_pr = pick_best_ae(step_size, model, x, x_adv_list, y)
    logits = model(final_pr)
    loss = F.cross_entropy(logits, y, reduction="mean")
    return loss, logits


def pick_best_ae(step_size, model, x, adv_list, y):
    max_distance = torch.zeros(y.size(0)).cuda()
    final_adv_example = adv_list[0] if adv_list else x
    for x_adv in adv_list:
        x_curr = x_adv.clone().detach()
        refine_lr = step_size
        x_curr = x_curr.requires_grad_(True)
        model.zero_grad()
        logits = model(x_curr)
        pred = logits.argmax(dim=1)
        is_ae = pred != y
        while is_ae.sum() > y.size(0) * 0.1:
            loss = F.cross_entropy(logits, y, reduction="mean")
            model.zero_grad()
            loss.backward()
            grad = x_curr.grad.detach()

            x_curr.data[is_ae] = x_curr.data[is_ae] - refine_lr * grad.data[is_ae].sign()
            x_curr.data[is_ae] = torch.clamp(x_curr.data[is_ae], 0, 1)

            x_curr = x_curr.detach().clone().requires_grad_(True)
            logits = model(x_curr)
            pred = logits.argmax(dim=1)
            is_ae = pred != y

        distance = torch.norm((x_adv - x_curr).view(x_adv.size(0), -1), dim=1, p=float('inf'))
        final_adv_example[distance>max_distance] = x_adv[distance>max_distance]
        max_distance[distance>max_distance] = distance[distance>max_distance]
    return final_adv_example



def ALP(model, x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10):
    x_pgd = pgd_loss(model, x, y, optimizer=optimizer,
                         step_size=step_size, 
                         epsilon=epsilon, 
                         attack_steps=attack_steps, 
                         attack=True)
    optimizer.zero_grad()
    logits = model(x_pgd) 
    robust_loss = F.cross_entropy(logits, y, reduction="mean")
    logit_diff = model(x_pgd) - model(x)
    logit_pairing_loss = torch.norm(logit_diff, dim=1).mean()
    total_loss = robust_loss + 0.01 * logit_pairing_loss
    # total_loss = robust_loss + logit_pairing_loss
    return total_loss, logits


def CLP(model, x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10):
    x_pgd = pgd_loss(model, x, y, optimizer=optimizer,
                         step_size=step_size, 
                         epsilon=epsilon, 
                         attack_steps=attack_steps, 
                         attack=True)
    optimizer.zero_grad()
    logits = model(x) 
    clean_loss = F.cross_entropy(logits, y, reduction="mean")
    logit_diff = model(x_pgd) - model(x)
    logit_pairing_loss = torch.norm(logit_diff, dim=1).mean()
    total_loss = clean_loss +  0.3 * logit_pairing_loss
    # total_loss = clean_loss +  logit_pairing_loss
    return total_loss, logits






import torch
import torchvision
import matplotlib.pyplot as plt
from cifar_model import *
from attack import *
from config import *
from src.train import Trainer
# from src.train_dog import Trainer
from src.train_tiny import Trainer
from src.eval import Evaluator




def main():
    args = parse_args()
    configs = get_configs(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = WRN(depth=configs.model_depth, width=configs.model_width, num_classes=configs.num_class)
    model = resnet("resnet18", configs.input_size, num_classes=configs.num_class, pretrained=False)
    # from torchvision import models
    # model = models.resnet18(pretrained=True)  
    # model.fc = nn.Linear(model.fc.in_features, configs.num_class)
    # model.to(device)


    if configs.mode == 'train':
        train = Trainer(configs, model)
        train.train_model()
    elif configs.mode == 'eval':
        test = Evaluator(configs, model)
        test.eval_model()
    else:
        raise ValueError('Specify the mode, `train` or `eval`')


if __name__ == '__main__':
    main()
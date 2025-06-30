import torch
import torchvision
import torchvision.transforms as transforms

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def data_prepare(args):

    transform_train =transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    transform_test =transforms.Compose([
        transforms.ToTensor()
    ])


    if args.dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform_train)
            ,batch_size=args.batch_size, shuffle=True, num_workers=8
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
            ,batch_size=args.batch_size, shuffle=False, num_workers=8
        )
    
    elif args.dataset == "CIFAR100":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=transform_train)
            ,batch_size=args.batch_size, shuffle=True, num_workers=8
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=transform_test)
            ,batch_size=args.batch_size, shuffle=False, num_workers=8
        )
    
    if args.dataset == "svhn":

        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root=args.data_root, split='train', download=True, transform=transform_train)
            , batch_size=args.batch_size, shuffle=True, num_workers=8
        )


        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root=args.data_root, split='test', download=True, transform=transform_test)
            , batch_size=args.batch_size, shuffle=False, num_workers=8
        )

    
    return train_loader, test_loader








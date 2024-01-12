import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []

    print(f'Using {args.cross_modal.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.trainer.optimizer.lr
        weight_decay = args.trainer.optimizer.weight_decay

        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.trainer.optimizer.lr * args.cross_modal.lr_factor # default 5.0
        if "bias" in key:
            lr = args.trainer.optimizer.lr * args.trainer.optimizer.bias_lr_factor
            weight_decay = args.trainer.optimizer.weight_decay_bias
        if "classifier" in key or "mlm_head" in key:
            lr = args.trainer.optimizer.lr * args.cross_modal.lr_factor
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = None
    if args.trainer.optimizer.opt == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.trainer.optimizer.lr, momentum=args.trainer.optimizer.momentum
        )
    elif args.trainer.optimizer.opt == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.trainer.optimizer.lr,
            betas=(args.trainer.optimizer.alpha, args.trainer.optimizer.beta),
            eps=1e-3,
        )
    elif args.trainer.optimizer.opt == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.trainer.optimizer.lr,
            betas=(args.trainer.optimizer.alpha, args.trainer.optimizer.beta),
            eps=1e-8,
        )
    else:
        raise NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.trainer.scheduler.milestones,
        gamma=args.trainer.scheduler.gamma,
        warmup_factor=args.trainer.scheduler.warmup_factor,
        warmup_epochs=args.trainer.scheduler.warmup_epochs,
        warmup_method=args.trainer.scheduler.warmup_method,
        total_epochs=args.trainer.num_epoch,
        mode=args.trainer.scheduler.lrscheduler,
        target_lr=args.trainer.scheduler.target_lr,
        power=args.trainer.scheduler.power,
    )

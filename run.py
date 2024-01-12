import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train, do_inference
from utils.checkpoint import Checkpointer
from utils import save_train_configs, load_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator

from utils.comm import get_rank, synchronize
import argparse
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def test(config_file):
    args = load_train_configs(config_file)

    args.training = False
    logger = setup_logger('DANK!1910', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UET Person search Args")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--test", action="store_true")
    #add params to optimize
    parser.add_argument("--l-names", nargs='+', default=[], type=str)
    parser.add_argument("--l-mlm-prob", type=float, default=None)
    parser.add_argument("--l-mlm-use-custom", action="store_true")
    parser.add_argument("--l-mim-prob", type=float, default=None)
    parser.add_argument("--l-mim-hog-bins", type=int, default=None)
    parser.add_argument("--l-mim-hog-pool", type=int, default=None)
    parser.add_argument("--l-triplet-k", type=int, default=None)
    parser.add_argument("--l-triplet-m-internal", type=int, default=None)
    parser.add_argument("--l-triplet-m-external", type=int, default=None)
    parser.add_argument("--l-triplet-weights", nargs='+', default=[], type=int)



    parser.add_argument("--lossweight-mim", type=float, default=None)
    parser.add_argument("--lossweight-mlm", type=float, default=None)
    parser.add_argument("--lossweight-triplet", type=float, default=None)

    

    args = parser.parse_args()
    if args.test: test(args.cfg)
    else: #TRAINING
        cfg = OmegaConf.load(args.cfg)
        set_seed(123)
        if len(args.l_names)    > 0: 
            print("loss is use = ", args.l_names)
            cfg.losses.loss_names = args.l_names
        cfg.losses.mlm.use_custom = args.l_mlm_use_custom
        if not (args.l_mlm_prob is None) : 
            print("[!!!]change MLM Prob")
            cfg.losses.mlm.mask_prob = args.l_mlm_prob

        if not args.l_mim_prob is None: 
            print("[!!!]change MIM Prob")
            cfg.losses.mim.mask_prob = args.l_mim_prob
        if not args.l_mim_hog_bins is None: 
            print("[!!!]change MIM HOG Bins")
            cfg.losses.mim.hog.bins = args.l_mim_hog_bins
        if not args.l_mim_hog_pool is None: 
            print("[!!!]change MIM HOG pool")
            cfg.losses.mim.hog.pool = args.l_mim_hog_pool

        if 'kntriplet' in args.l_names: cfg.dataloader.sampler = 'identity'
        if not args.l_triplet_k is None: 
            print("[!!!]change TRIPLET K")
            cfg.losses.kntriplet = args.l_triplet_k
        if not args.l_triplet_m_internal is None: 
            print("[!!!]change TRIPLET margin of internal pair ")
            cfg.losses.kntriplet.i2im = cfg.losses.kntriplet.t2tm=args.l_triplet_m_internal
        if not args.l_triplet_m_external is None: 
            print("[!!!]change TRIPLET margin of external pair ")
            cfg.losses.kntriplet.i2tm = cfg.losses.kntriplet.t2tm=args.l_triplet_m_external
        if len(args.l_triplet_weights) > 0: 
            print("[!!!]change TRIPLET triplet weights")
            cfg.losses.kntriplet.weights = args.l_triplet_weights

        if not args.lossweight_mim is None: 
            print("[!!!]change \\lamda of MIM ")
            cfg.losses.mim_loss_weight = args.lossweight_mim
        if not args.lossweight_mlm is None: 
            print("[!!!]change \\lamda of MLM ")
            cfg.losses.mlm_loss_weight = args.lossweight_mlm
        if not args.lossweight_triplet is None: 
            print("[!!!]change \\lamda of TRIPLET ")
            cfg.losses.kntriplet_loss_weight = args.lossweight_triplet



        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        cfg.distributed = num_gpus > 1

        if cfg.distributed:
            torch.cuda.set_device(cfg.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()
        
        device = "cuda"
        cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        cfg.output_dir = output_dir = op.join(cfg.iocfg.savedir, cfg.dataloader.dataset_name, f'{cur_time}_{cfg.name}')
        logger = setup_logger('DANK!1910', save_dir=output_dir, if_train=True, distributed_rank=get_rank())
        logger.info("Using {} GPUs".format(num_gpus))
        save_train_configs(output_dir, cfg)

        # get image-text pair datasets dataloader
        train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(cfg)
        # for idx, loader in enumerate([train_loader]):
        #     for x in loader:
        #         print(f"\n========{idx}============")
        #         print(x)
        #         break
        
        
        # Build models
        model = build_model(cfg, num_classes)
        logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        model.to(device)
        

        if cfg.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[cfg.distribution_cfg.local_rank],
                output_device=cfg.distribution_cfg.local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
            )
        else: model = torch.nn.DataParallel(model)
        optimizer = build_optimizer(cfg, model)
        scheduler = build_lr_scheduler(cfg, optimizer)

        is_master = get_rank() == 0
        checkpointer = Checkpointer(model, optimizer, scheduler, output_dir, is_master)
        evaluator = Evaluator(val_img_loader, val_txt_loader)

        start_epoch = 1
        if cfg.resume:
            checkpoint = checkpointer.resume(cfg.resume_ckpt_file)
            start_epoch = checkpoint['epoch']
        do_train(start_epoch, cfg, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
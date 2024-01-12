import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import torchvision.transforms as T



def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.trainer.log_period
    eval_period = args.trainer.eval_period
    device = "cuda"
    num_epoch = args.trainer.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("DANK!1910.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),

        "mim_loss": AverageMeter(),
        
        'ritc_loss': AverageMeter(),
        'citc_loss': AverageMeter(),
        "triplet_loss": AverageMeter(),
        
        'intra_distil_loss':AverageMeter(),
        'inter_distil_loss':AverageMeter(),

        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    args.cur_step = 0
    stpe = len(train_loader)  #step per epoch
    args.total_step = num_epoch * stpe
    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values(): meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            args.cur_step =  (epoch-1) * stpe + n_iter + 1 
            batch = {k: v.to(device) for k, v in batch.items()}
            ret = model(batch)
            
            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # synchronize()
            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['mim_loss'].update(ret.get('mim_loss', 0), batch_size)
            meters['citc_loss'].update(ret.get('citc_loss', 0), batch_size)
            meters['ritc_loss'].update(ret.get('ritc_loss', 0), batch_size)
            # meters['triplet_loss'].update(ret.get('triplet_loss', 0), batch_size)

            meters['intra_distil_loss'].update(ret.get('intra_distil_loss', 0), batch_size)
            meters['inter_distil_loss'].update(ret.get('inter_distil_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)
            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    # if v.avg > 0:
                        info_str += f", {k}: {v.avg:.2f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
            
        scheduler.step()
        
        #use tensorboard to log
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,train_loader.batch_size / time_per_batch))
        if epoch % eval_period  == 0 :
            if get_rank() == 0:
                
                if args.distributed or isinstance(model, torch.nn.DataParallel):
                    top1,table, top1_ema, table_ema = evaluator.eval(model.module.eval(), i2t_metric=True, return_table=True, return_ema=args.ema.enable)
                else:
                    top1, table, top1_ema, table_ema = evaluator.eval(model.eval(), i2t_metric=True, return_table=True, return_ema=args.ema.enable)
                logger.info("Validation Results - Epoch: {} - Top1={} \n ".format(epoch, top1) + str(table) + "\n\n")
                if args.ema.enable: logger.info("\t EMA Results - Epoch: {} - Top1={} \n ".format(epoch, top1_ema) + str(table_ema) + "\n\n")
                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
                if args.ema.enable and best_top1 < top1_ema:
                    best_top1 = top1_ema
                    arguments["epoch"] = epoch
                    checkpointer.save("best", ema=True, **arguments)

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("DANK!1910.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())

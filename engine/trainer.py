import time
import torch
import wandb
from torch.autograd import Variable
from utils.util import get_grad_norm, get_total_gpu_memory


def train_epoch(args, data_loader, data_loader_t, model, step_per_epoch,
                optimizer, criterion, scheduler, epoch, logger,
                FL, CE, entropyKD):
    
    all_loss, epoch_all_loss = 0, 0
    reg_loss, epoch_reg_loss = 0, 0
    cls_loss, epoch_cls_loss = 0, 0
    dist_loss, epoch_dist_loss = 0, 0
    adv_loss, epoch_adv_loss = 0, 0
     
    start_time = time.time()

    batch_iterator = iter(data_loader)
    batch_iterator_t = iter(data_loader_t)
    for i, iteration in enumerate(range(1, step_per_epoch+1)):     
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        
        try:
            images_t, targets_t = next(batch_iterator_t)
        except StopIteration:
            batch_iterator_t = iter(data_loader_t)
            images_t, targets_t = next(batch_iterator_t)


        images = Variable(images.to(args.device, non_blocking=True))
        targets = [Variable(ann.to(args.device, non_blocking=True)) for ann in targets]

        images_t = Variable(images_t.to(args.device, non_blocking=True))
        targets_t = [Variable(ann.to(args.device, non_blocking=True)) for ann in targets_t]
        
        optimizer.zero_grad()
        
        out, loss_dist, loss_adv = model(images, images_t, "train", CE, FL, entropyKD)

        loss_dist = loss_dist * args.alpha
        loss_adv = loss_adv * args.adv

        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c + loss_adv + loss_dist

        try:
            loss.backward()
        except:
            print("LOSS BACKWARD ERRRORRRR")
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        total_norm = get_grad_norm(model.parameters())

        optimizer.step()
        scheduler.step()
        
        get_lr = scheduler.get_last_lr()[0]
        get_moment = scheduler.optimizer.param_groups[0]['momentum']

        all_loss += loss.item()
        reg_loss += loss_l.item()
        cls_loss += loss_c.item()
        dist_loss += loss_dist
        adv_loss += loss_adv.item()
        epoch_all_loss += loss.item()
        epoch_reg_loss += loss_l.item()
        epoch_cls_loss += loss_c.item()
        epoch_dist_loss += loss_dist
        epoch_adv_loss += loss_adv.item()

        wandb.log({"train/grad_norm": total_norm})
        wandb.log({"scheduler/lr": get_lr})
        wandb.log({"scheduler/momentum": get_moment})

        wandb.log({"train/loss": loss.item()})
        wandb.log({"train/cls_loss": loss_c.item()})
        wandb.log({"train/reg_loss": loss_l.item()})
        wandb.log({"train/dist_loss": loss_dist})
        wandb.log({"train/local_loss": loss_adv.item()})
        

        if iteration % args.disp_interval == 0:
            all_loss /= args.disp_interval
            reg_loss /= args.disp_interval
            cls_loss /= args.disp_interval
            dist_loss /= args.disp_interval
            adv_loss /= args.disp_interval
            
            end_time = time.time()
            
            cons_gpu, total_gpu = get_total_gpu_memory()
            
            logger.info('[epoch %2d][iter %4d/%4d]|| Loss: %.4f || lr: %.2e || grad_norm: %.2f  || reg_loss: %.4f || cls_loss: %.4f || dist_loss: %.4f || adv_loss: %.4f || Time: %.2f sec || VRAM: %.2f/%.2f GB' \
                % (epoch, iteration, step_per_epoch, all_loss, get_lr, total_norm, reg_loss, cls_loss, dist_loss, adv_loss, end_time - start_time, cons_gpu, total_gpu))

            all_loss = 0
            reg_loss = 0
            cls_loss = 0
            dist_loss = 0
            adv_loss = 0

            start_time = time.time()

    logger.info("Train Epoch %2d || Loss: %.4f || cls_loss: %.4f || reg_loss: %.4f || dist_loss: %.4f || adv_loss: %.4f" % (epoch, epoch_all_loss / i, epoch_cls_loss / i, \
                                                                                        epoch_reg_loss / i, epoch_dist_loss / i, epoch_adv_loss/i))


def warm_up_lr(optimizer, cur_step, max_step):
    if cur_step == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / max_step
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * cur_step /(cur_step - 1)

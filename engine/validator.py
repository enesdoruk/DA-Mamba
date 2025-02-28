import wandb
import torch
from torch.autograd import Variable


def validate_epoch(args, data_loader, step_per_epoch, model, criterion, epoch, logger):
    all_loss = 0
    reg_loss = 0
    cls_loss = 0
    
    model.eval()

    batch_iterator = iter(data_loader)
    with torch.no_grad():
        for i, iteration in enumerate(range(1, step_per_epoch+1)):
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                images, targets = next(batch_iterator)

            images = Variable(images.to(args.device, non_blocking=True))
            targets = [Variable(ann.to(args.device, non_blocking=True)) for ann in targets]

            out, _, _ = model(images, None, "train")

            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c

            all_loss += loss.item()
            reg_loss += loss_l.item()
            cls_loss += loss_c.item()

            wandb.log({"val/loss": loss.item()})
            wandb.log({"val/cls_loss": loss_c.item()})
            wandb.log({"val/reg_loss": loss_l.item()})

    logger.info("Val Epoch %2d || Loss: %.4f || cls_loss: %.4f || reg_loss: %.4f" % (epoch, all_loss/i, cls_loss/i, reg_loss/i))
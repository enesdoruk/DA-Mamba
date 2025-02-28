import warnings
warnings.filterwarnings("ignore")


import os
import time
import torch
import wandb
import argparse
import torch.utils
import torch.utils.data
import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable

from data import *
from data.config import *
from eval import test_net
from utils.logger import create_logger
from models.ssd import build_ssd
from models.layers.modules import MultiBoxLoss
from utils.augmentations import SSDAugmentation
from utils.util import get_grad_norm, weights_init
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule
from utils.visualization import tsne_visualization



parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC',
                    choices=['VOC', 'CS'],
                    type=str)
parser.add_argument('--dataset_target', default='clipart',
                    choices=['clipart', 'water', 'comic', 'fogcs'],
                    type=str)
parser.add_argument('--dataset_root', default='/AI/adaptEdNet/datasets/VOCdevkit',
                    help='Dataset root directory path')
parser.add_argument('--dataset_target_root', default='/AI/adaptEdNet/datasets/clipart',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--decay_type', default='cosine',
                    help='Scheduler type')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument("--max_grad_norm", default=20.0, type=float, help="Max gradient norm.")
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--end_epoch', default=50, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/train/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pretrained_dir', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--disp_interval', default=100, type=int,
                    help='Number of iterations to display')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for.")
parser.add_argument('--wandb_name', default='trailer', type=str,
                    help='run name')


def train(args, logger):
    if args.dataset == 'VOC':
        from data import VOCDetection
        from data import VOC_CLASSES as labelmap
        cfg = voc
        set_type = 'test'
        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        testset = VOCDetection(args.dataset_root, [('2007', set_type)], BaseTransform(300, MEANS))
    elif args.dataset == 'CS':
        from data import CSDetection
        from data import CS_CLASSES as labelmap
        cfg = cs
        dataset = CSDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))

    if args.dataset_target == 'clipart':
        from data import CLPDetection
        from data import CLP_CLASSES as labelmap_t
        set_type_t = 'test'
        dataset_t = CLPDetection(root=args.dataset_target_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        testset_t = CLPDetection(args.dataset_target_root, [('2007', set_type_t)], BaseTransform(300, MEANS))
    elif args.dataset_target == 'water':
        from data import WATERDetection
        from data import WATER_CLASSES as labelmap_t
        set_type_t = 'test'
        dataset_t = WATERDetection(root=args.dataset_target_root,transform=SSDAugmentation(cfg['min_dim'], MEANS))
        testset_t = WATERDetection(args.dataset_target_root, [('2007', set_type_t)], BaseTransform(300, MEANS))
    elif args.dataset_target == 'comic':
        from data import COMICDetection
        from data import COMIC_CLASSES as labelmap_t
        set_type_t = 'test'
        dataset_t = COMICDetection(root=args.dataset_target_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        testset_t = COMICDetection(args.dataset_target_root, [('2007', set_type_t)], BaseTransform(300, MEANS))
    elif args.dataset_target == 'fogcs':
        from data import CSFGDetection
        from data import CSFG_CLASSES as labelmap_t
        set_type_t = 'test'
        dataset_t = CSFGDetection(root=args.dataset_target_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        testset_t = CSFGDetection(args.dataset_target_root, [('2007', set_type_t)], BaseTransform(300, MEANS))


    model = build_ssd(cfg['min_dim'], cfg['num_classes'])

    wandb.watch(model)

    if args.resume:
        logger.info('Resuming training, loading {}...'.format(args.resume))
        model.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.pretrained_dir + args.basenet)
        logger.info('Loading base network...')
        model.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        model = model.to(device='cuda', non_blocking=True)

    if not args.resume:
        logger.info('Initializing weights...')
        model.extras.apply(weights_init)
        model.loc.apply(weights_init)
        model.conf.apply(weights_init)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    
    t_total = (len(dataset) // args.batch_size) * (args.end_epoch - args.start_epoch)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    

    model.train()

    step_per_epoch = len(dataset) // args.batch_size

    logger.info('The number of dataset %s is %d' % (args.dataset, len(dataset)))
    logger.info('Using the specified args:')
    logger.info(args)
    logger.info('Loading the dataset...')

    data_loader = torch.utils.data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  pin_memory=False,
                                  collate_fn=detection_collate,
                                  )
    
    step_per_epoch = len(dataset) // args.batch_size

    for epoch in range(args.start_epoch, args.end_epoch+1):
        model.train()

        epoch_time = time.time()

        all_loss = 0
        reg_loss = 0
        cls_loss = 0

        start_time = time.time()
        batch_iterator = iter(data_loader)
        for iteration in range(1, step_per_epoch+1):
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                images, targets = next(batch_iterator)

            optimizer.zero_grad()

            images = Variable(images.cuda())
            targets = [Variable(ann.cuda()) for ann in targets]

            out, feats = model(images, "train")

            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            total_norm = get_grad_norm(model.parameters())

            wandb.log({"train/grad_norm": total_norm})

            optimizer.step()
            scheduler.step()

            get_lr = scheduler.get_last_lr()[0]
            get_moment = scheduler.optimizer.param_groups[0]['momentum']

            wandb.log({"scheduler/lr": get_lr})
            wandb.log({"scheduler/momentum": get_moment})

            all_loss += loss.item()
            reg_loss += loss_l.item()
            cls_loss += loss_c.item()
            
            if iteration % args.disp_interval == 0:
                all_loss /= args.disp_interval
                reg_loss /= args.disp_interval
                cls_loss /= args.disp_interval
               
                det_loss = reg_loss+cls_loss
                end_time = time.time()

                
                logger.info('[epoch %2d][iter %4d/%4d]|| Loss: %.4f || lr: %.2e || grad_norm: %.2f  || det_loss: %.4f || reg_loss: %.4f || cls_loss: %.4f || Time: %.2f sec' \
                      % (epoch, iteration, step_per_epoch, all_loss, get_lr, total_norm, det_loss, reg_loss, cls_loss, end_time - start_time))
              
                
                wandb.log({"train/loss": det_loss})
                wandb.log({"train/cls_loss": cls_loss})
                wandb.log({"train/reg_loss": reg_loss})

                all_loss = 0
                reg_loss = 0
                cls_loss = 0
                start_time = time.time()
                

        logger.info('This epoch cost %.4f sec'%(time.time()-epoch_time))
        
        if (epoch+1) % 5 == 0:
            model.eval()

            logger.info("---------------------- EVALUATION FOR TARGET ----------------------")
            annopath = os.path.join(args.dataset_target_root, 'VOC2007', 'Annotations', '%s.xml') 
            devkit_path = args.save_folder + args.dataset_target + f"_{datetime.now().hour}-{datetime.now().minute}"
            imgsetpath = os.path.join(args.dataset_target_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
            _ = test_net(annopath, imgsetpath, labelmap_t, args.save_folder, model, args.cuda, testset_t,
                    BaseTransform(300, MEANS), args.top_k, 300,
                    thresh=args.confidence_threshold, phase='test', set_type=set_type_t, devkit_path=devkit_path, logger=logger)
           

        if (epoch+1) % 10 == 0 :
            save_pth = os.path.join(args.save_folder, str(epoch)+'.pth')
            torch.save(model.state_dict(), save_pth)


    model.eval()

    logger.info("---------------------- EVALUATION FOR SOURCE ----------------------")
    annopath = os.path.join(args.dataset_root, 'VOC2007', 'Annotations', '%s.xml') 
    devkit_path = args.save_folder + args.dataset + f"_{datetime.now().hour}-{datetime.now().minute}"
    imgsetpath = os.path.join(args.dataset_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
    source_tsne = test_net(annopath, devkit_path, imgsetpath,  args.save_folder, model, args.cuda, testset,
             BaseTransform(300, MEANS), args.top_k, 300, thresh=args.confidence_threshold,
             phase='test', set_type=set_type_t, devkit_path=devkit_path, logger=logger)


    logger.info("---------------------- EVALUATION FOR TARGET ----------------------")
    annopath = os.path.join(args.dataset_target_root, 'VOC2007', 'Annotations', '%s.xml') 
    devkit_path = args.save_folder + args.dataset_target + f"_{datetime.now().hour}-{datetime.now().minute}"
    imgsetpath = os.path.join(args.dataset_target_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
    target_tsne = test_net(annopath, devkit_path, imgsetpath, args.save_folder, model, args.cuda, testset_t,
             BaseTransform(300, MEANS), args.top_k, 300, thresh=args.confidence_threshold,
             phase='test', set_type=set_type_t, devkit_path=devkit_path, logger=logger)


    tsne_visualization(target_tsne, source_tsne, iter='final_tsne')


if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logger = create_logger(output_dir='logs', name=f"AdaptEdNet{str(datetime.today().strftime('_%d-%m-%H'))}")

    wandb.init(project="AdaptEd-Net", name=f"{args.wandb_name}{str(datetime.today().strftime('_%d-%m-%H'))}")

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            logger.warning("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)


    train(args, logger)
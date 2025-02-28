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
import torch.backends.cudnn as cudnn
from torch.utils.data import RandomSampler

from data import *
from eval import test_net
from engine.trainer import train_epoch
from utils.logger import create_logger
from engine.davimnet import build_davimnet
from engine.validator import validate_epoch
from utils.augmentations import SSDAugmentation
from utils.visualization import tsne_visualization
from engine.models.layers.modules import MultiBoxLoss
from utils.util import weights_init, mamba_init_weights
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule
from utils.lossZoo import FocalLoss, CrossEntropyLoss, EntropyKD


device = torch.device('cuda:0')

        
def eval(args, dataset_root, dataset, logger, model, testset, set_type, labelmap):
    model.eval()

    annopath = os.path.join(dataset_root, 'VOC2007', 'Annotations', '%s.xml') 
    devkit_path = args.save_folder + dataset + f"_{datetime.now().hour}-{datetime.now().minute}"
    save_folder = args.save_folder + dataset + f"_{datetime.now().hour}-{datetime.now().minute}"
    imgsetpath = os.path.join(dataset_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
    tsne = test_net(annopath, imgsetpath, labelmap, save_folder, model, args.device, testset,
                    BaseTransform(args.size, MEANS), args.top_k, args.size,
                    thresh=args.confidence_threshold, phase='test', set_type=set_type, devkit_path=devkit_path, logger=logger)
    return tsne, save_folder


def train(args, logger, cfg, dataset, testset, set_type_t, dataset_t, testset_t):
    model = build_davimnet(cfg['min_dim'], cfg['num_classes'])

    cudnn.benchmark = True
    
    model = model.to(args.device, non_blocking=True)
        
    logger.info('Initializing weights...')

    model.extras.apply(mamba_init_weights)
    model.loc.apply(weights_init)
    model.conf.apply(weights_init)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    FL = FocalLoss(num_classes=2, gamma=args.gamma_fl)
    CE = CrossEntropyLoss(num_classes=2)
    entropyKD = EntropyKD(T=1)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.device)
    
    t_total = (len(dataset) // args.batch_size) * (args.end_epoch - args.start_epoch)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    
    logger.info('The number of dataset %s is %d' % (args.dataset, len(dataset)))

    train_sampler, train_sampler_t = RandomSampler(dataset), RandomSampler(dataset_t)

    data_loader = torch.utils.data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  drop_last=True,
                                  sampler=train_sampler,
                                  persistent_workers=True,
                                  pin_memory=True,
                                  collate_fn=detection_collate,
                                  )
    
    data_loader_t = torch.utils.data.DataLoader(dataset_t, args.batch_size,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  sampler=train_sampler_t,
                                  shuffle=False,
                                  persistent_workers=True,
                                  pin_memory=True,
                                  collate_fn=detection_collate,
                                  )
    
    # data_loader_test = torch.utils.data.DataLoader(testset_t, args.batch_size_val,
    #                               num_workers=args.num_workers,
    #                               drop_last=True,
    #                               shuffle=False,
    #                               pin_memory=True,
    #                               persistent_workers=True,
    #                               collate_fn=detection_collate,
    #                               )
    
    step_per_epoch = len(dataset) // args.batch_size
    step_per_epoch_val = len(testset_t) // args.batch_size_val

    torch.cuda.empty_cache()

    model.zero_grad()
    for epoch in range(args.start_epoch, args.end_epoch+1):
        model.train()
        
        epoch_time = time.time()
        
        train_epoch(args, data_loader, data_loader_t, model, step_per_epoch,
                    optimizer, criterion, scheduler, epoch, logger,
                    FL, CE, entropyKD)
        
        # validate_epoch(args, data_loader_test, step_per_epoch_val, model, criterion, epoch, logger)

        logger.info('This epoch cost %.4f sec'%(time.time()-epoch_time))
        logger.info("="*50)
        
        if (epoch+1) % 10 == 0:
            logger.info("---------------------- EVALUATION FOR TARGET ----------------------")
            
            _, save_folder = eval(args, args.dataset_target_root, args.dataset_target, logger, model, testset_t, set_type_t, labelmap_t)

            save_pth = os.path.join(save_folder, str(epoch)+'.pth')
            torch.save(model.state_dict(), save_pth)

    logger.info("---------------------- EVALUATION FOR SOURCE ----------------------")
    source_tsne, _ = eval(args, args.dataset_root, args.dataset, logger, model, testset_t, set_type, labelmap_t)

    logger.info("---------------------- EVALUATION FOR TARGET ----------------------")
    target_tsne, _ = eval(args, args.dataset_target_root, args.dataset_target, logger, model, testset_t, set_type_t, labelmap_t)

    tsne_visualization(target_tsne, source_tsne, iter='final_tsne')


def train_args():
    parser = argparse.ArgumentParser(description='DAVimNet Training With Pytorch')
    parser.add_argument('--dataset', default='VOC',choices=['VOC'],type=str)
    parser.add_argument('--dataset_target', default='clipart',choices=['clipart', 'water', 'comic'],type=str)
    parser.add_argument('--dataset_root', default='/cta/users/undergrad3/Desktop/DAVimNet/datasets/VOCdevkit',help='Dataset root directory path')
    parser.add_argument('--dataset_target_root', default='/cta/users/undergrad3/Desktop/DAVimNet/datasets/clipart',help='Dataset root directory path')
    parser.add_argument('--decay_type', default='cosine',help='Scheduler type')
    parser.add_argument('--batch_size', default=10, type=int,help='Batch size for training')
    parser.add_argument('--batch_size_val', default=4, type=int,help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,help='Checkpoint state_dict file to resume training from')
    parser.add_argument("--max_grad_norm", default=20.0, type=float, help="Max gradient norm.")
    parser.add_argument('--start_epoch', default=0, type=int,help='Resume training at this iter')
    parser.add_argument('--end_epoch', default=50, type=int,help='Resume training at this iter')
    parser.add_argument('--num_workers', default=0, type=int,help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool,help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='weights/train/',help='Directory for saving checkpoint models')
    parser.add_argument('--pretrained_dir', default='weights/',help='Directory for saving checkpoint models')
    parser.add_argument('--disp_interval', default=100, type=int,help='Number of iterations to display')
    parser.add_argument('--confidence_threshold', default=0.01, type=float,help='Detection confidence threshold')
    parser.add_argument('--top_k', default=5, type=int,help='Further restrict the number of predictions to parse')
    parser.add_argument('--size', default=224, type=int,help='image size')
    parser.add_argument('--warmup_steps', default=500, type=int, help='Step of training to perform learning rate warmup for.')
    parser.add_argument('--wandb_name', default='trailer', type=str,help='run name')
    parser.add_argument('--alpha', default=1.0, type=float, help='dist loss value')
    parser.add_argument('--adv', default=1.0, type=float, help='dist loss value')
    parser.add_argument('--gamma_fl', default=5, type=float,help='Gamma for focal loss')
    return parser.parse_args()


if __name__ == '__main__':
    args = train_args()

    if args.dataset == 'VOC':
        from data import VOCDetection
        cfg = voc_davimnet
        set_type = 'test'
        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        testset = VOCDetection(args.dataset_root, [('2007', set_type)], BaseTransform(args.size, MEANS))
    elif args.dataset == 'cs':
        from data import CSDetection
        cfg = cs_davimnet
        dataset = CSDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))

    if args.dataset_target == 'clipart':
        from data import CLPDetection
        from data import CLP_CLASSES as labelmap_t
        set_type_t = 'test'
        dataset_t = CLPDetection(root=args.dataset_target_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        testset_t = CLPDetection(args.dataset_target_root, [('2007', set_type_t)], BaseTransform(args.size, MEANS))
    elif args.dataset_target == 'water':
        from data import WATERDetection
        from data import WATER_CLASSES as labelmap_t
        set_type_t = 'test'
        dataset_t = WATERDetection(root=args.dataset_target_root,transform=SSDAugmentation(cfg['min_dim'], MEANS))
        testset_t = WATERDetection(args.dataset_target_root, [('2007', set_type_t)], BaseTransform(args.size, MEANS))
    elif args.dataset_target == 'comic':
        from data import COMICDetection
        from data import COMIC_CLASSES as labelmap_t
        set_type_t = 'test'
        dataset_t = COMICDetection(root=args.dataset_target_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        testset_t = COMICDetection(args.dataset_target_root, [('2007', set_type_t)], BaseTransform(args.size, MEANS))
    elif args.dataset_target == 'csfog':
        from data import CSFGDetection
        from data import CSFG_CLASSES as labelmap_t
        set_type_t = 'test'
        dataset_t = COMICDetection(root=args.dataset_target_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        testset_t = COMICDetection(args.dataset_target_root, [('2007', set_type_t)], BaseTransform(args.size, MEANS))

    args.device = torch.device("cuda:0")

    os.makedirs('logs', exist_ok=True)
    logger = create_logger(output_dir='logs', name=f"DAVimNet{str(datetime.today().strftime('_%d-%m-%H'))}")

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    
    logger.info('Using the specified args:')
    logger.info(args)
    logger.info('Loading the dataset...')

    wandb.init(project="DAVimNet", name=f"{args.wandb_name}{str(datetime.today().strftime('_%d-%m-%H'))}")

    train(args, logger, cfg, dataset, testset, set_type_t, dataset_t, testset_t)



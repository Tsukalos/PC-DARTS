import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from ucf101dct import UCF101DCTDataset, UCF101DCTDataset_MV

from torch.autograd import Variable
from model import NetworkVideo as Network
from model import get_mv_model


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='ucf101', help='location of the data corpus')
parser.add_argument('--fold', type=int, default=1, choices=[1,2,3], help='fold to use for trainning/eval')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=5e-3, help='init learning rate')
parser.add_argument('--mv_learning_rate', type=float, default=1e-3, help='init mv learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-1, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=25, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.75, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=8, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.4, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=25, help='random seed')
parser.add_argument('--arch', type=str, default='PC_UCF_e', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--fbs_channels', type=int, default=32, choices=[16,32,64], help='Frequency band selection. The number of dct channels used.')

args = parser.parse_args()

root_save_dir = '{}/{}'.format(args.model_path, args.set)
args.save = '{}/eval-{}-{}'.format(root_save_dir,
                                   args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

VIDEO_CLASSES = 101
if args.set == 'hmdb51':
    VIDEO_CLASSES = 51


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    torch.set_num_threads(3)

    dct_coefficients = args.fbs_channels

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, VIDEO_CLASSES, args.layers,
                    args.auxiliary, genotype, dct_channels=dct_coefficients)
    
    mv_model = get_mv_model(1, VIDEO_CLASSES)
    
    model = model.cuda()
    mv_model = mv_model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.learning_rate,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay
    # )

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        amsgrad=True,
        weight_decay=args.weight_decay
    )

    mv_optimizer = torch.optim.SGD(
        mv_model.parameters(),
        args.mv_learning_rate,
        momentum=args.momentum,
        weight_decay=3e-4
    )

    train_transform, valid_transform = utils._data_transforms_ucf101_train(args)
    mv_train_transform, mv_valid_transform = utils._data_transforms_mv(args)

    if args.set == 'ucf101':
        train_data = UCF101DCTDataset_MV("../data/UCF101jpg_/UCF-101",
                                      "../data/UCFDCTSplits_jpg/UCFjpg-list_.txt",
                                      "../data/UCFDCTSplits_jpg/",
                                      "../data/UCFDCTSplits_jpg/UCFmv-list_.txt",
                                      transform=train_transform,
                                      mv_transform=mv_train_transform,
                                      num_coeff=dct_coefficients, seed=args.seed,
                                      train=True, shuffle=True, normalize_I_frames=True)
        valid_data = UCF101DCTDataset_MV("../data/UCF101jpg_/UCF-101",
                                      "../data/UCFDCTSplits_jpg/UCFjpg-list_.txt",
                                      "../data/UCFDCTSplits_jpg/",
                                      "../data/UCFDCTSplits_jpg/UCFmv-list_.txt",
                                      transform=valid_transform,
                                      mv_transform=mv_valid_transform,
                                      num_coeff=dct_coefficients, seed=args.seed,
                                      train=False, shuffle=True, normalize_I_frames=True)
    else:
        logging.info(f'set {args.set} not found')
        sys.exit(1)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=2)
    
    # mean = 0.0
    # for images, _, _ in train_queue:
    #     batch_samples = images.size(0) 
    #     images = images.view(batch_samples, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    # mean = mean / len(train_queue.dataset)

    # var = 0.0
    # pixel_count = 0
    # for images,_, _ in train_queue:
    #     batch_samples = images.size(0)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    #     pixel_count += images.nelement() / images.size(1)
    # std = torch.sqrt(var / pixel_count)

    # print(mean, std)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs))
    mv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        mv_optimizer, float(args.epochs))
    
    best_acc = 0.0
    for epoch in range(args.epochs):

        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        logging.info('epoch %d mv_lr %e', epoch, mv_scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        
        train_acc, train_obj = train(train_queue, model, mv_model, criterion, optimizer, mv_optimizer)
        mv_scheduler.step()
        scheduler.step()
        logging.info('train_acc %f, train_loss %f', train_acc, train_obj)

        torch.cuda.empty_cache()

        valid_acc, valid_obj = infer(valid_queue, model, mv_model, criterion)
        if valid_acc > best_acc:
            best_acc = valid_acc
        logging.info('valid_acc %f, valid_loss %f, best_acc %f', valid_acc, valid_obj, best_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        utils.save(mv_model, os.path.join(args.save, 'mv_weights.pt'))


def train(train_queue, model, mv_model, criterion, optimizer, mv_optimizer):
    objs = utils.AvgrageMeter()
    objs_I = utils.AvgrageMeter()
    objs_MV = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input_list_I, input_list_MV, target) in enumerate(train_queue):
        model.train()
        assert input_list_MV.size(0) == input_list_I.size(0)

        late_loss = torch.tensor(np.zeros(1)).cuda()
        late_logits = torch.tensor(np.zeros((input_list_MV.size(0), VIDEO_CLASSES))).cuda()
        acu_loss_I = torch.tensor(np.zeros(1)).cuda()
        acu_loss_MV = torch.tensor(np.zeros(1)).cuda()
        acu_logits_I = torch.tensor(np.zeros((input_list_I.size(0), VIDEO_CLASSES))).cuda()
        acu_logits_MV = torch.tensor(np.zeros((input_list_MV.size(0), VIDEO_CLASSES))).cuda()
        acu_logits_I_aux = torch.tensor(np.zeros((input_list_I.size(0), VIDEO_CLASSES))).cuda()

        frames_list = input_list_I.transpose(1,0)
        mvs_list = input_list_MV.transpose(1,0)

        optimizer.zero_grad()
        mv_optimizer.zero_grad()
        for input_I, input_MV in zip(frames_list, mvs_list):
            n = input_I.size(0)
            input_I = Variable(input_I).cuda()
            input_MV = Variable(input_MV).cuda()
            target = Variable(target).cuda(non_blocking=True)
            logits, logits_aux = model(input_I)
            # loss = criterion(logits, target)
            
            # acu_loss_I += loss
            acu_logits_I += logits
            acu_logits_I_aux += logits

            logits_ = mv_model(input_MV)
            # loss_ = criterion(logits_, target)

            # acu_loss_MV += loss_
            acu_logits_MV += logits_

        

        acu_logits_I = acu_logits_I/len(frames_list)
        # acu_loss_I = acu_loss_I/len(frames_list)

        acu_logits_MV = acu_logits_MV/len(mvs_list)
        # acu_loss_MV = acu_loss_MV/len(mvs_list)

        acu_loss_I = criterion(acu_logits_I, target)
        acu_loss_MV = criterion(acu_logits_MV, target)
        if args.auxiliary:
            loss_aux = criterion(acu_logits_I_aux, target)
            acu_loss_I += args.auxiliary_weight*loss_aux

        late_loss = (acu_loss_I*0.75+acu_loss_MV*0.25)
        late_logits = (acu_logits_I*0.75+acu_logits_MV*0.25)

        late_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        mv_optimizer.step()

        prec1, prec5 = utils.accuracy(late_logits, target, topk=(1, 5))
        objs.update(late_loss.data.item(), n)
        objs_I.update(acu_loss_I.data.item(), n)
        objs_MV.update(acu_loss_MV.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f %e %e', step, objs.avg, top1.avg, top5.avg, objs_I.avg, objs_MV.avg)
            
        acu_loss_I = acu_logits_I = input_I = None
        acu_loss_MV = acu_logits_MV = input_MV = None
        late_loss = late_logits = None

        torch.cuda.empty_cache()

    return top1.avg, objs.avg


def infer(valid_queue, model, mv_model, criterion):
    objs = utils.AvgrageMeter()
    objs_I = utils.AvgrageMeter()
    objs_MV = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input_list_I, input_list_MV, target) in enumerate(valid_queue):
            model.eval()
            assert input_list_MV.size(0) == input_list_I.size(0)

            late_loss = torch.tensor(np.zeros(1)).cuda()
            acu_loss_I = torch.tensor(np.zeros(1)).cuda()
            acu_loss_MV = torch.tensor(np.zeros(1)).cuda()

            late_logits = torch.tensor(np.zeros((input_list_MV.size(0), VIDEO_CLASSES))).cuda()
            acu_logits_I = torch.tensor(np.zeros((input_list_I.size(0), VIDEO_CLASSES))).cuda()
            acu_logits_MV = torch.tensor(np.zeros((input_list_MV.size(0), VIDEO_CLASSES))).cuda()
            acu_logits_I_aux = torch.tensor(np.zeros((input_list_I.size(0), VIDEO_CLASSES))).cuda()

            frames_list = input_list_I.transpose(1,0)
            mvs_list = input_list_MV.transpose(1,0)

            for input_I, input_MV in zip(frames_list, mvs_list):
                n = input_I.size(0)
                input_I = Variable(input_I).cuda()
                input_MV = Variable(input_MV).cuda()
                target = Variable(target).cuda(non_blocking=True)
                logits, _ = model(input_I)
                # loss = criterion(logits, target)
                
                # acu_loss_I += loss
                acu_logits_I += logits
                acu_logits_I_aux += logits

                logits_ = mv_model(input_MV)
                # loss_ = criterion(logits_, target)

                # acu_loss_MV += loss_
                acu_logits_MV += logits_

            

            acu_logits_I = acu_logits_I/len(frames_list)
            # acu_loss_I = acu_loss_I/len(frames_list)

            acu_logits_MV = acu_logits_MV/len(mvs_list)
            # acu_loss_MV = acu_loss_MV/len(mvs_list)

            acu_loss_I = criterion(acu_logits_I, target)
            acu_loss_MV = criterion(acu_logits_MV, target)
            if args.auxiliary:
                loss_aux = criterion(acu_logits_I_aux, target)
                acu_loss_I += args.auxiliary_weight*loss_aux

            late_loss = (acu_loss_I*0.75+acu_loss_MV*0.25)
            late_logits = (acu_logits_I*0.75+acu_logits_MV*0.25)

            prec1, prec5 = utils.accuracy(late_logits, target, topk=(1, 5))
            n = input_I.size(0)
            objs.update(late_loss.data.item(), n)
            objs_I.update(acu_loss_I.data.item()*0.5, n)
            objs_MV.update(acu_loss_MV.data.item()*0.5, n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f %e %e', step, objs.avg, top1.avg, top5.avg, objs_I.avg, objs_MV.avg)
                
            acu_loss_I = acu_logits_I = input_I = None
            acu_loss_MV = acu_logits_MV = input_MV = None
            late_loss = late_logits = None

            torch.cuda.empty_cache()

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

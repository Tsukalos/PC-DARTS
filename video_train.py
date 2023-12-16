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
from ucf101dct import UCF101DCTDataset

from torch.autograd import Variable
from model import NetworkVideo as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--set', type=str, default='ucf101',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true',
                    default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float,
                    default=0.5, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.4, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--arch', type=str, default='PC_UCF_f1_e40_step6',
                    help='which architecture to use')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--fbs_channels', type=int, default=32, choices=[16,32,64],
                    help='Frequency band selection. The number of dct channels used.')

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
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.learning_rate,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay
    # )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        args.learning_rate,
        eps=0.0001,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_ucf101_train(None)
    if args.set == 'ucf101':
        train_data = UCF101DCTDataset("../data/UCF101jpg/UCF-101",
                                      "../data/UCFDCTSplits_jpg/UCFjpg-list.txt",
                                      "../data/UCFDCTSplits_jpg/",
                                      transform=train_transform,
                                      num_coeff=dct_coefficients, seed=args.seed,
                                      train=True, shuffle=True)
        valid_data = UCF101DCTDataset("../data/UCF101jpg/UCF-101",
                                      "../data/UCFDCTSplits_jpg/UCFjpg-list.txt",
                                      "../data/UCFDCTSplits_jpg/",
                                      transform=valid_transform,
                                      num_coeff=dct_coefficients, seed=args.seed,
                                      train=False, shuffle=True)
    else:
        logging.info(f'set {args.set} not found')
        sys.exit(1)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs))
    best_acc = 0.0
    for epoch in range(args.epochs):

        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        scheduler.step()
        logging.info('train_acc %f', train_acc)

        torch.cuda.empty_cache()

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        if valid_acc > best_acc:
            best_acc = valid_acc
        logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input_list, target) in enumerate(train_queue):
        model.train()

        acu_loss = torch.tensor(np.zeros(1)).cuda()
        acu_logits = torch.tensor(
            np.zeros((input_list.size(0), VIDEO_CLASSES))).cuda()
        frames_list = input_list.transpose(1,0)
        optimizer.zero_grad()
        for input in frames_list:
            n = input.size(0)
            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(
                non_blocking=True)

            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += args.auxiliary_weight*loss_aux
            acu_loss += loss
            acu_logits += logits

        acu_logits = acu_logits/len(frames_list)
        acu_loss = acu_loss/len(frames_list)
        acu_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(acu_logits, target, topk=(1, 5))
        objs.update(acu_loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        acu_loss = None
        acu_logits = None
        input = None
        torch.cuda.empty_cache()

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step,
                         objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input_list, target) in enumerate(valid_queue):
            acu_loss = torch.tensor(np.zeros(1)).cuda()
            acu_logits = torch.tensor(
                np.zeros((input_list.size(0), VIDEO_CLASSES))).cuda()
            frames_list = input_list.transpose(1,0)
            for input in frames_list:
                input = Variable(input).cuda()
                target = Variable(target).cuda(non_blocking=True)
                logits, __ = model(input)
                loss = criterion(logits, target)
                acu_logits += logits
                acu_loss += loss

            acu_logits = acu_logits/len(frames_list)
            acu_loss = acu_loss/len(frames_list)

            prec1, prec5 = utils.accuracy(acu_logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(acu_loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            acu_loss = None
            acu_logits = None
            input = None
            torch.cuda.empty_cache()

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step,
                             objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

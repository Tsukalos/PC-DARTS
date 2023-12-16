import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
from ucf101dct import UCF101DCTDataset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkVideo as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int,
                    default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='eeeee', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true',
                    default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--arch', type=str, default='PC_DARTS_UCF_f1_e20',
                    help='which architecture to use')
parser.add_argument('--fbs_channels', type=int, default=32,
                    help='Frequency band selection. The number of dct channels used.')

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

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

    dct_coefficients = 32
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels,
                    VIDEO_CLASSES,
                    args.layers,
                    args.auxiliary,
                    genotype,
                    dct_channels=dct_coefficients)
    
    model = model.cuda()
    utils.load(model, args.model_path)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    _, test_transform = utils._data_transforms_ucf101(None)
    test_data = UCF101DCTDataset("../data/UCF101jpg/UCF-101",
                                 "../data/UCFDCTSplits/UCFjpg-list.txt",
                                 "../data/UCFDCTSplits/",
                                 transform=test_transform,
                                 num_coeff=dct_coefficients, train=False, seed=args.seed)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = args.drop_path_prob
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input_list, target) in enumerate(test_queue):
        # input = input.cuda()
        # target = target.cuda(non_blocking=True)
        predicts = torch.tensor(
            np.zeros((input_list[0].size(0), VIDEO_CLASSES))).cuda()
        for input in input_list:
            input = Variable(input).cuda()
            target = Variable(target).cuda(non_blocking=True)
            logits = model(input)
            loss = criterion(logits, target)
            predicts += logits

        max_logits = predicts/len(input_list)

        prec1, prec5 = utils.accuracy(max_logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step,
                         objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

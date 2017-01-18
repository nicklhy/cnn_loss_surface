"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, fit
from common.data import get_mnist_iter
import mxnet as mx
import numpy as np


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network         = 'mlp',
        # train
        gpus            = None,
        batch_size      = 64,
        disp_batches    = 20,
        num_epochs      = 20,
        model_period    = 20,
        lr              = .05,
        lr_step_epochs  = '10',
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbol.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, get_mnist_iter)

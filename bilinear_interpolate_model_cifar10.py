import os
import argparse
import cPickle
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data
from common.util import download_file
import mxnet as mx
import numpy as np

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

def download_cifar10():
    data_dir="data"
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    return fnames

def load_params(filename):
    params = {
        'arg_params': {},
        'aux_params': {},
    }

    for k,v in mx.nd.load(filename).items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            params['arg_params'][name] = v
        elif tp == 'aux':
            params['aux_params'][name] = v

    return params


if __name__ == '__main__':
    # download data
    (train_fname, val_fname) = download_cifar10()

    parser = argparse.ArgumentParser(description="interpolate score",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--net-json', type=str, required=True,
                        help='symbol\'s json file')
    parser.add_argument('--params1', type=str, required=True,
                        help='the first network weights file')
    parser.add_argument('--params2', type=str, required=True,
                        help='the second network weights file')
    parser.add_argument('--params3', type=str, required=True,
                        help='the second network weights file')
    parser.add_argument('--params4', type=str, required=True,
                        help='the second network weights file')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='the batch size')
    parser.add_argument('--alpha1', type=float, default=0.0,
                       help='lower bound of alpha')
    parser.add_argument('--alpha2', type=float, default=1.0,
                       help='higher bound of alpha')
    parser.add_argument('--alpha-num', type=int, default=20,
                       help='number of alpha values')
    parser.add_argument('--beta-num', type=int, default=20,
                       help='number of beta values')
    parser.add_argument('--gpus', type=str,
                        help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    parser.add_argument('--output', type=str,
                        help='output file')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='key-value store type')
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 10,
        num_examples  = 50000,
        image_shape    = '3,28,28',
        pad_size       = 4,
        # train
        batch_size     = 128,
    )
    args = parser.parse_args()

    params1 = load_params(args.params1)
    params2 = load_params(args.params2)
    params3 = load_params(args.params3)
    params4 = load_params(args.params4)

    kv = mx.kvstore.create(args.kv_store)
    train, val = data.get_rec_iter(args, kv)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # create model
    model = mx.mod.Module(
        context       = devs,
        symbol        = mx.sym.load(args.net_json)
    )
    model.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)

    # evaluation metrices
    eval_metrics = mx.metric.create('accuracy')

    alpha_list = np.linspace(args.alpha1, args.alpha2, args.alpha_num+1)
    beta_list = np.linspace(args.alpha1, args.alpha2, args.beta_num+1)
    alpha_grid, beta_grid = np.meshgrid(alpha_list, beta_list)
    train_error_grid = np.zeros(alpha_grid.shape)
    val_error_grid = np.zeros(alpha_grid.shape)
    for i in range(alpha_grid.shape[0]):
        for j in range(alpha_grid.shape[1]):
            alpha = alpha_grid[i, j]
            beta = beta_grid[i, j]
            #  compute tmp weights
            tmp_params = {
                'arg_params': {},
                'aux_params': {},
            }
            for name in tmp_params.keys():
                for k, theta_0 in params1[name].items():
                    theta_1 = params2[name][k]
                    theta_2 = params3[name][k]
                    theta_3 = params4[name][k]
                    tmp_params[name][k] = (1-beta)*((1-alpha)*theta_0+alpha*theta_1)+beta*((1-alpha)*theta_2+alpha*theta_3)

            #  set tmp weights
            model.set_params(arg_params=tmp_params['arg_params'],
                             aux_params=tmp_params['aux_params'])

            print 'alpha = %f, beta = %f ...' % (alpha, beta)

            if train is not None:
                model.score(train, eval_metrics, reset=True)
                train_error_grid[i, j] = 1-eval_metrics.get()[1]
                print '\ttrain error = %f' % train_error_grid[i, j]
            if val is not None:
                model.score(val, eval_metrics, reset=True)
                val_error_grid[i, j] = 1-eval_metrics.get()[1]
                print '\tval error = %f' % val_error_grid[i, j]

    if args.output is None:
        args.output = os.path.join(ROOT_DIR,
                                   'cache',
                                   os.path.split(args.params1)[-1].replace('.params', '')+'_'+os.path.split(args.params2)[-1].replace('.params', '')+'_'+os.path.split(args.params3)[-1].replace('.params', '')+'_'+os.path.split(args.params4)[-1].replace('.params', '')+'_bilinear_interpolate.pkl')
        if not os.path.exists(os.path.dirname(args.output)):
            os.mkdir(os.path.dirname(args.output))
    with open(args.output, 'wb') as fd:
        cPickle.dump({
            'alpha_grid': alpha_grid,
            'beta_grid': beta_grid,
            'train_error_grid': train_error_grid,
            'val_error_grid': val_error_grid,
        }, fd)
        print 'save log to %s' % (args.output)

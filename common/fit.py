import mxnet as mx
import logging
import os
import time

def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = args.begin_epoch
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

def _load_model(args, rank=0):
    if 'params' not in args or args.params is None:
        return (None, None, None)
    if args.net_json is not None:
        sym = mx.sym.load(args.net_json)
    else:
        sym = None
    arg_params = {}
    aux_params = {}
    for k,v in mx.nd.load(args.params).items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        elif tp == 'aux':
            aux_params[name] = v

    logging.info('Loaded model %s', args.params)
    return (sym, arg_params, aux_params)

def _save_model(mod, args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.module_checkpoint(mod, args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank), period=args.model_period)

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--net-json', type=str,
                       help='network json file')
    train.add_argument('--params', type=str,
                       help='pretrained weights')
    train.add_argument('--begin-epoch', type=int, default=0,
                       help='begin epoch')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd and nag')
    train.add_argument('--rho', type=float, default=0.9,
                       help='rho for AdaDelta')
    train.add_argument('--beta1', type=float, default=0.9,
                       help='beta1 for Adam')
    train.add_argument('--beta2', type=float, default=0.999,
                       help='beta2 for Adam')
    train.add_argument('--gamma1', type=float, default=0.95,
                       help='gamma1 for RMSProp')
    train.add_argument('--gamma2', type=float, default=0.9,
                       help='gamma2 for RMSProp')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    train.add_argument('--model-period', type=int, default=1,
                       help='model snapshot period')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--test-io', type=int, default=0,
                       help='1 means test reading speed without training')
    return train

def fit(args, network, data_loader, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    # data iterators
    (train, val) = data_loader(args, kv)
    if args.test_io:
        tic = time.time()
        for i, batch in enumerate(train):
            for j in batch.data:
                j.wait_to_read()
            if (i+1) % args.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, args.disp_batches*args.batch_size/(time.time()-tic)))
                tic = time.time()

        return


    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    # create model
    model = mx.mod.Module(
        context       = devs,
        symbol        = network
    )

    lr_scheduler  = lr_scheduler
    optimizer_params = {
            'learning_rate': lr,
            'wd' : args.wd,
            'lr_scheduler': lr_scheduler}
    if args.optimizer.lower() == 'sgd' or args.optimizer.lower() == 'nag':
        optimizer_params['momentum'] = args.mom
    elif args.optimizer.lower() == 'adam':
        optimizer_params['beta1'] = args.beta1
        optimizer_params['beta2'] = args.beta2
    elif args.optimizer.lower() == 'rmsprop':
        optimizer_params['gamma1'] = args.gamma1
        optimizer_params['gamma2'] = args.gamma2
    if args.optimizer.lower() == 'adadelta':
        optimizer_params['rho'] = args.rho

    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None

    # save model
    checkpoint = _save_model(model, args, kv.rank)

    initializer   = mx.init.Xavier(
       rnd_type='gaussian', factor_type="in", magnitude=2)
    # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),
    model.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
    model.init_params(initializer=initializer,
                      arg_params=arg_params,
                      aux_params=aux_params)
    #  save the initialized weights
    arg_params, aux_params = model.get_params()
    if 'params' not in args or args.params is None:
        checkpoint(-1, model.symbol, arg_params, aux_params)

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

    # run
    model.fit(train,
        begin_epoch        = args.begin_epoch,
        num_epoch          = args.num_epochs,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        optimizer          = args.optimizer,
        optimizer_params   = optimizer_params,
        #  initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True,
        monitor            = monitor)

## AN EMPIRICAL ANALYSIS OF DEEP NETWORK LOSS SURFACES WITH MXNET
----------------
A MXNet implementation of [AN EMPIRICAL ANALYSIS OF DEEP NETWORK LOSS SURFACES](https://arxiv.org/abs/1612.04010).

**Important: This code can not generate absolutely the same visualization results with the original paper yet.**

### Usage
1. To train a ResNet-32 network on cifar10 with adam, run the following command:
```
$ python train_cifar10.py --model-prefix models/cifar10/resnet32_adam --network resnet --num-layers 32 --optimizer adam --model-period 100 --params models/cifar10/resnet32_sgd-0000.params --gpus 0
```
2. To compute the train/val error curve(50 points) between two different models:
```
$ python linear_interpolate_model_cifar10.py --net-json models/cifar10/resnet_sgd-symbol.json --params1 models/cifar10/resnet_sgd-0300.params --params2 models/cifar10/resnet_adam-0300.params --alpha-num 50 --batch-size 16 --alpha-num 50 --gpus 0
```
3. To compute the train/val error mesh(alpha num = 50, beta num = 50) among three different models using the barycentric interpolate method:
```
$ python barycentric_interpolate_model_cifar10.py --net-json models/cifar10/resnet_sgd-symbol.json --params1 models/cifar10/resnet_rmsprop-0300.params --params2 models/cifar10/resnet_adam-0300.params --params3 models/cifar10/resnet_sgd-0300.params --alpha-num 50 --beta-num 50 --gpus 0
```
4. To compute the train/val error mesh(alpha num = 50, beta num = 50) among four different models using the bilinear interpolate method:
```
$ python bilinear_interpolate_model_cifar10.py --net-json models/cifar10/resnet_sgd-symbol.json --params1 models/cifar10/resnet_adam-0000.params --params2 models/cifar10/resnet_sgd-0300.params --params3 models/cifar10/resnet_adam-0300.params --params4 models/cifar10/resnet_rmsprop-0300.params --alpha-num 50 --beta-num 50 --gpus 0
```
5. To plot the train/val error curve between two models, please use visualize\_interpolate\_2d.ipynb.

6. To plot the train/val error surface among three or four models, please use visualize\_interpolate\_3d.ipynb.

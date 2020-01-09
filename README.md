This repository is forked from [dcgan_vae_pytorch](https://github.com/seangal/dcgan_vae_pytorch/).
I study this by reading and commenting.

# dcgan_vae_pytorch
dcgan combined with vae in pytorch!

this code is based on [pytorch/examples](https://github.com/pytorch/examples) and [staturecrane/dcgan_vae_torch](https://github.com/staturecrane/dcgan_vae_torch)

The original artical can be found [here](https://arxiv.org/abs/1512.09300)
## Requirements
* torch
* torchvision
* visdom
* (optional) lmdb

## Usage
to start visdom:
```
python -m visdom.server
```


to start the training:
```
usage: main.py [-h] --dataset DATASET --dataroot DATAROOT [--workers WORKERS]
               [--batchSize BATCHSIZE] [--imageSize IMAGESIZE] [--nz NZ]
               [--ngf NGF] [--ndf NDF] [--niter NITER] [--saveInt SAVEINT] [--lr LR]
               [--beta1 BETA1] [--cuda] [--ngpu NGPU] [--netG NETG]
               [--netD NETD]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | imagenet | folder | lfw
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF
  --ndf NDF
  --niter NITER         number of epochs to train for
  --saveInt SAVEINT     number of epochs between checkpoints
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --cuda                enables cuda
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
```

_ここからは追加分_

## dataset

-[Body-parts dataset list](https://github.com/arXivTimes/arXivTimes/blob/a51bfe64a3862e4cceafa3863b777f473f6c0900/datasets/README.md#bodyparts)
-[IMDb-face](https://github.com/fwang91/IMDb-Face)
-[IMDb-face download script](https://github.com/IrvingShu/IMDb-Face-Download)

### aditional info - Asian facial image dataset

[Can anyone help me find a database with asian faces?](https://www.researchgate.net/post/Can_anyone_help_me_find_a_database_with_asian_faces)
- [The Asian Face Age Dataset (AFAD)](https://afad-dataset.github.io/)
- [CASIA-FaceV5](http://biometrics.idealtest.org/dbDetailForUser.do?id=9)

## requirements

- pytorch
- torchvision
- argparse
- visdom
visdomはrequest2.2.0が必要なので環境を作る時に注意すること

それでも問題が起こるようならばcondaをupdateする
[RemoveError: 'pyopenssl' is a dependency of conda and cannot be removed from conda's operating environment.が出た時にやったこと メモ](https://knaka20blue.hatenablog.com/entry/2019/02/25/131937)
conda update --force conda

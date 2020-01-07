from __future__ import print_function

# https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0
# argument Parser
import argparse

import os
import random
import math
import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# visdom is visualize the process of learning
# torchsummary is network summarizing tool
# both are usefull!
# https://qiita.com/yasudadesu/items/1dda5f9d1708b6d4d923
import visdom

from torch.autograd import Variable

vis = visdom.Visdom()
vis.env = 'vae_dcgan'

# define argument parser
# usage detail:
# https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--saveInt', type=int, default=25, help='number of epochs between checkpoints')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

# optでparserの解析を実施する
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf) # 引数に渡されたディレクトリに出力imageとモデルのチェックポイントを書き込むディレクトリを作る
except OSError:
    pass # もしエラーが発生した場合には何もしない

# 引数にmanualSeedが渡された場合の処理
# https://qiita.com/chat-flip/items/4c0b71a7c0f5f6ae437f
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000) # manualSeedがNoneの場合、1~10000の間の一様乱数から整数を取得してseedにする
print("Random Seed: ", opt.manualSeed) # 設定されたseedをprint
random.seed(opt.manualSeed) # 乱数のseedにopt.manualSeedを設定
torch.manual_seed(opt.manualSeed) # 畳み込み層の重みの乱数seedを設定
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed) # GPUを使う場合は別の設定にする

# オートチューナが最適なアルゴリズムを見つける
# https://qiita.com/koshian2/items/9877ed4fb3716eac0c37
cudnn.benchmark = True

# GPUが利用可能な環境だがopt.cuda引数がtrueでないときにwarningを出す処理
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
# 画像データセットが以下のときの処理
# 画像データセット毎に異なる前処理を実施している
if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
# cifar10の場合はcenterCropをしていない
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset # datasetが空でないかどうかをチェックする https://qiita.com/nannoki/items/15004992b6bb5637a9cd

# dataloaderの定義、より詳細はpytorchのチュートリアルを見ること
# https://qiita.com/takurooo/items/e4c91c5d78059f92e76d
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# argument parserから受け取った引数の値の一部をグローバル変数に格納
ngpu = int(opt.ngpu) # GPUの数
nz = int(opt.nz) # 潜在変数のベクトル長 デフォルトは100
ngf = int(opt.ngf) # デフォルトは64
ndf = int(opt.ndf) # デフォルトは64
nc = 3 # チャンネル数は3で固定

# pytorchのConv2dについて
# Conv2d(インプットのチャンネル数,アウトプットのチャンネル数,カーネルサイズ)
# https://qiita.com/kazetof/items/6a72926b9f8cd44c218e#43-nnconv2d%E3%81%A8%E3%81%AF

# custom weights initialization called on netG and netD
# ネットワークの重みを初期化する関数の定義
# netGならびにnetDに対して、applyで渡される関数であり、それぞれのネットワークのサブモジュールに対して実施される
# https://tutorialmore.com/questions-56844.htm
def weights_init(m):
    classname = m.__class__.__name__ # 定義されたクラスの元のクラス名を取得する　https://ja.stackoverflow.com/questions/4556/%E3%83%A1%E3%83%B3%E3%83%90%E9%96%A2%E6%95%B0%E3%81%8B%E3%82%89%E3%82%AF%E3%83%A9%E3%82%B9%E3%81%AE%E5%90%8D%E5%89%8D%E3%82%92%E5%8F%96%E5%BE%97%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95
    if classname.find('Conv') != -1: # classnameが含まれる場合の処理（.findで-1が返ってくると指定文字列は含まれていない） https://www.sejuku.net/blog/52207
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# 潜在変数にGaussianを仮定し、そこからのサンプリング用のclass
# VAEは潜在空間にGaussianを仮定するため
# https://qiita.com/kenmatsu4/items/b029d697e9995d93aa24
# この部分がReparameterization Trickに該当する
# つまり、学習させるのは正規乱数のパラメータの方である
class _Sampler(nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()
        
    def forward(self,input):
        # inputは2要素のリストであり、index=0が平均値ベクトル、index=1が分散の対数となっている
        # Samplerはencoderの出力の結果である2つのベクトルを引数としてとるため、
        # 一方を平均ベクトル、もう一方を対数分散ベクトルとなるよう、重みをトレーニングする形となる
        mu = input[0]
        logvar = input[1] # 分散の対数
        
        std = logvar.mul(0.5).exp_() #calculate the STDEV
        if opt.cuda:
            # epsにstdのサイズと同じサイズのtensorを作る
            # tensorの中身にはnormal_()でmean=0,std=1の正規分布に従う乱数を格納する
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.normal_
            eps = torch.cuda.FloatTensor(std.size()).normal_() #random normalized noise
        else:
            # 上と同様
            eps = torch.FloatTensor(std.size()).normal_() #random normalized noise
        eps = Variable(eps) # 自動微分を有効にするためVariableクラスでラップする必要があったが、0.4以降は特にいらない？　https://codezine.jp/article/detail/11052
        # 戻り値は標準正規分布乱数からmu,stdのパラメータに従う正規分布に戻したもの
        return eps.mul(std).add_(mu) 
    
# encoderの定義
class _Encoder(nn.Module):
    # 初期化関数
    # 引数に画像サイズをとる
    def __init__(self,imageSize):
        super(_Encoder, self).__init__()
        # 2を底とするXの対数を返す
        # 2^n = imageSize なので、imageSizeは2のx乗でなければならない
        # また、少なくとも3回は畳み込むのでimageSizeが8(2^3)でなければならない
        n = math.log2(imageSize)
        
        # 条件に合致しないときの処理
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        # nを整数型に変換する
        n=int(n)

        # pytorchのConv2dについて
        # Conv2d(インプットのチャンネル数,アウトプットのチャンネル数,カーネルサイズ)
        # https://qiita.com/kazetof/items/6a72926b9f8cd44c218e#43-nnconv2d%E3%81%A8%E3%81%AF
        # more detail
        # https://pytorch.org/docs/stable/nn.html
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        # ここではself.encoderの出力結果をチャネル数nzにして返す畳み込み層を定義している
        # forwardで二つのvectorを返している
        self.conv1 = nn.Conv2d(ngf * 2**(n-3), nz, 4) # (channel of Input, channel of output, kernel size)
        self.conv2 = nn.Conv2d(ngf * 2**(n-3), nz, 4) # (channel of Input, channel of output, kernel size)

        # encoderを畳み込み層-バッチノーマライゼーション-LeakyReLUの繰り返しで定義
        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        # https://pytorch.org/docs/stable/nn.html
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # 入力チャネル数=3, outputチャネル数=ngf, カーネルサイス=4, ストライド2, パディング=1
        # 出力サイズは ((H(W)+2P-Fh(Fn))/2)+1で計算できる
        # 入力サイズが64x64の時、出力サイズは32x32になる
        # 出力チャネル数はngfで定義（デフォルトで64）
        self.encoder.add_module('input-conv',nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        # 活性化関数にLeakyReLUを使用、x<0のときの傾きを0.2に固定
        self.encoder.add_module('input-relu',nn.LeakyReLU(0.2, inplace=True))
        # 画像サイズが64x64のとき、64=2^5なのでn=6
        # その場合、0～2(6-3=3なので)の繰り返しとなる
        for i in range(n-3):
            # state size. (ngf) x 32 x 32
            # モジュールの名称をpyramid(入力チャネル数)-(出力チャネル数)としている
            # 入力チャネル数および出力チャネル数は記載の通り、ngfの2^i倍ずつ増えていく
            # カーネルサイズ、ストライド、パディングはずっと同じ、バイアスは使用しない
            self.encoder.add_module('pyramid.{0}-{1}.conv'.format(ngf*2**i, ngf * 2**(i+1)), nn.Conv2d(ngf*2**(i), ngf * 2**(i+1), 4, 2, 1, bias=False))
            self.encoder.add_module('pyramid.{0}.batchnorm'.format(ngf * 2**(i+1)), nn.BatchNorm2d(ngf * 2**(i+1)))
            self.encoder.add_module('pyramid.{0}.relu'.format(ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))
            # モジュールの繰り返しごとに画像サイズが半分になっていく
            # 最終的に4x4の画像になるようにnが調整されている
            # 最終的なチャネル数は2^(n-3)である。画像サイズ64x64の場合はngf*8となる

        # state size. (ngf*8) x 4 x 4

    # encoderで入力画像を(ngf*8) x 4 x 4に畳み込み
    # その出力をconv1,conv2にかけてnzのサイズの二つのベクトルに変換しreturn
    def forward(self,input):
        output = self.encoder(input)
        return [self.conv1(output),self.conv2(output)]

# Generative Model全体の定義
# decoder部分はencoderの逆を行う
class _netG(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        # すでに定義したEncoderとSamplerを定義
        self.encoder = _Encoder(imageSize)
        self.sampler = _Sampler()
        
        # encoderと同様に繰り返し回数nはimageSize=2^nとしたときのnとする
        # decoderではencoderの逆で倍々にupsamplingしていく
        n = math.log2(imageSize)
        
        # imageSizeが規定に満たないときの処理
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        # forループを回すためにnを整数型に直す
        n=int(n)

        # decoderをencoderの逆として定義する
        # まず最初にサイズnzのベクトルを入力値に取り、４ｘ４画像に拡大する
        self.decoder = nn.Sequential()
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2**(n-3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2**(n-3)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        # 入力チャネル数、出力チャネル数はencoderの逆
        # forループ内のコードはencoderと同じだが、rangeの定義が異なる
        for i in range(n-3, 0, -1):
            self.decoder.add_module('pyramid.{0}-{1}.conv'.format(ngf*2**i, ngf * 2**(i-1)),nn.ConvTranspose2d(ngf * 2**i, ngf * 2**(i-1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid.{0}.batchnorm'.format(ngf * 2**(i-1)), nn.BatchNorm2d(ngf * 2**(i-1)))
            self.decoder.add_module('pyramid.{0}.relu'.format(ngf * 2**(i-1)), nn.LeakyReLU(0.2, inplace=True))

        # 最終的に元のサイズに戻し、活性化関数tanhを通す
        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False))
        self.decoder.add_module('output-tanh', nn.Tanh())

    # 順方向
    # endocer-sampler-decoderの順に流して最終結果を得る
    def forward(self, input):
        # GPUが使える場合
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.sampler, output, range(self.ngpu))
            output = nn.parallel.data_parallel(self.decoder, output, range(self.ngpu))
        # GPUが使えない場合
        else:
            output = self.encoder(input)
            output = self.sampler(output)
            output = self.decoder(output)
        return output
    # GPUが使える場合に備えてencoder,sampler,decoderをcuda用に変換する関数
    # インスタンス化する際に登場する
    def make_cuda(self):
        self.encoder.cuda()
        self.sampler.cuda()
        self.decoder.cuda()

# 生成ネットワークの定義
netG = _netG(opt.imageSize,ngpu)
# applyはnetG内でselfで定義されたサブモジュールに対して(fn)の関数を適用する
# この場合はweights_initなので、重みの初期化を全サブモジュールに対して適用している
netG.apply(weights_init)
# すでに何らかの学習が行われておりnetGが空でない場合はすでにあるnetGから読み込みを行う
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Discriminatorモデルの定義
class _netD(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        # こちらも同じようにimageSize=2^nを定義する
        n = math.log2(imageSize)
        
        # 画像サイズが規定外だったときの処理
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)
        
        # 以下でモデルの定義を行う
        self.main = nn.Sequential()

        # input is (nc) x 64 x 64
        # ここから画像サイズが半分ずつになっていく
        # ndfはDiscriminator用の出力チャネル数
        # デフォルトの場合、おおもとの入力チャネル数nc=3、最初の出力チャネル数ndf=64
        self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ndf) x 32 x 32
        # 64x64の場合、n=6なので、0,1,2の繰り返し
        # 32x32-16x16-8x8-4x4となり、このループの中では画像サイズが4x4になる
        # ngfとなっているところはndfのtypo?
        for i in range(n-3):
            self.main.add_module('pyramid.{0}-{1}.conv'.format(ngf*2**(i), ngf * 2**(i+1)), nn.Conv2d(ndf * 2 ** (i), ndf * 2 ** (i+1), 4, 2, 1, bias=False))
            self.main.add_module('pyramid.{0}.batchnorm'.format(ngf * 2**(i+1)), nn.BatchNorm2d(ndf * 2 ** (i+1)))
            self.main.add_module('pyramid.{0}.relu'.format(ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))

        # 最後の層で出力チャネル数1、シグモイド関数にかける
        self.main.add_module('output-conv', nn.Conv2d(ndf * 2**(n-3), 1, 4, 1, 0, bias=False))
        self.main.add_module('output-sigmoid', nn.Sigmoid())
        

    # 順方向、書いてあるそのまま
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

# netGと同じように定義
netD = _netD(opt.imageSize,ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# 損失関数にはそれぞれBCELoss(Binary Cross Entropy Loss)とMSELoss(Mean Square Error)を使う
# https://pytorch.org/docs/stable/nn.html
# 必要におうじてもっと詳しく調べること
criterion = nn.BCELoss()
MSECriterion = nn.MSELoss()

# 入力値のサイズを設定
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
# ノイズのサイズを設定、ノイズはバッチサイズx潜在変数ベクトル長x1x1
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
# 固定ノイズをnoizeと同じサイズとし、標準正規分布からサンプリングする
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
# ラベルのサイズを設定、バッチサイズと同じ
label = torch.FloatTensor(opt.batchSize)
# 真であれば1、偽であれば0のラベルとする
real_label = 1
fake_label = 0

# GPUが使える場合はすでに設定したモデル、変数をcuda用に変換しておく
if opt.cuda:
    netD.cuda()
    netG.make_cuda()
    criterion.cuda()
    MSECriterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# 自動微分用にVariableクラスに変換
# 現在は必要ないか？
input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# setup optimizer
# 最適化法はnetD,netGいずれもAdamとする
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

gen_win = None
rec_win = None

# ここから学習部分
# 繰り返し回数はoptionで設定、デフォルトは25
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # 先にDiscriminatorを訓練する
        # data中の画像をreal、ノイズをfakeとしてrealが1になるように訓練していく
        
        # train with real
        # 勾配をゼロに初期化
        netD.zero_grad()
        # dataから画像をとってくる
        real_cpu, _ = data
        # 画像数をバッチサイズとする
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        # dataからとってきた画像のラベルをすべて1にする
        label.data.resize_(real_cpu.size(0)).fill_(real_label)

        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        gen = netG.decoder(noise)
        gen_win = vis.image(gen.data[0].cpu()*0.5+0.5,win = gen_win)
        label.data.fill_(fake_label)
        output = netD(gen.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()
        ############################
        # (2) Update G network: VAE
        ###########################
        
        netG.zero_grad()
        
        encoded = netG.encoder(input)
        mu = encoded[0]
        logvar = encoded[1]
        
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        
        sampled = netG.sampler(encoded)
        rec = netG.decoder(sampled)
        rec_win = vis.image(rec.data[0].cpu()*0.5+0.5,win = rec_win)
        
        MSEerr = MSECriterion(rec,input)
        
        VAEerr = KLD + MSEerr;
        VAEerr.backward()
        optimizerG.step()

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################

        label.data.fill_(real_label)  # fake labels are real for generator cost

        rec = netG(input) # this tensor is freed from mem at this point
        output = netD(rec)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 VAEerr.data[0], errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

    if epoch%opt.saveInt == 0 and epoch!=0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

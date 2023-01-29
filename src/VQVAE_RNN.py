from __future__ import print_function
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from six.moves import xrange
import torchvision
from tqdm import tqdm

from utils import get_args, get_system_info, set_figure_options, set_seed, configure_cudnn
from dataset import RandomFrameDataset, get_recon_dataset
from loss import LOSS_DICT
from optimizer import OPTIM_DICT
from random import randrange
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from PIL import Image
import random

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        ### Create an embedding matrix with size number of embedding X embedding dimension
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) # look up table
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        #print(f"at vqvae: {inputs.shape}")
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        # Calculate distances between flattened input and embedding vector
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        # Choose indices that are min in each row
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        ## Create a matrix of dimensions B*H*W into number of embeddings
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        ### Convert index to on hot encoding 
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        #print(f"at vqvae - quantized: {quantized.shape}")
        # Loss
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        #print(f"at vqvae - e_loss: {e_latent_loss}")
        q_latent_loss = nn.functional.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        #print(f"at vqvae - loss: {loss}")
        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

### Create Residual connections
class Residual(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_hiddens):
        super(Residual,self).__init__()
        self._block=nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                     out_channels=num_residual_hiddens,
                     kernel_size=3,stride=1,padding=1,bias=False), # channel: 128 -> 32
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                     out_channels=num_hiddens,
                     kernel_size=1,stride=1,bias=False) # channel: 32 -> 128
        )
        
    def forward(self,x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(ResidualStack,self).__init__()
        self._num_residual_layers=num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels,num_hiddens,num_residual_hiddens) for _ in range(self._num_residual_layers)])
            # _num_residual_layers(32개)만큼 Residual을 실행 (32개 layers)
    def forward(self,x):
        for i in range(self._num_residual_layers): # 32개의 layer들이 있고, 거기에 x를 넣는다.
            x=self._layers[i](x)
        return nn.functional.relu(x)

class Encoder(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(Encoder,self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens//2,
                                kernel_size=4,
                                stride=2,padding=1) # channel: 3 -> 64
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels = num_hiddens,
                                 kernel_size=4,
                                 stride=2,padding=1
                                ) # channel: 64 -> 128
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=3, #원래 3,1,1
                                stride=2,padding=1) # channel: 128 -> 128
        self._residual_stack = ResidualStack(in_channels = num_hiddens,
                                             num_hiddens = num_hiddens,
                                             num_residual_layers = num_residual_layers,
                                             num_residual_hiddens = num_residual_hiddens
                                            )
    def forward(self,inputs):
        #print(f"0st: {inputs.shape}")
        x = self._conv_1(inputs) #inputs.cpu()에서 에러떠서 바꿈.
        x = nn.functional.relu(x)
        #print(f"1st: {x.shape}")

        x = self._conv_2(x)
        #print(f"2nd: {x.shape}")
        x = nn.functional.relu(x)

        x = self._conv_3(x)
        #x = nn.functional.leaky_relu(x)
        #print(f"3rd: {x.shape}")
        
        #x = self._conv_4(x)
        x = self._residual_stack(x)
        #print(f"encoder final:{x.shape}")
        return x

class Decoder(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(Decoder,self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels= num_hiddens,
                                kernel_size=3,
                                stride=1,padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens= num_residual_hiddens
                                            )
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                               out_channels=num_hiddens//2,
                                               kernel_size=4,
                                               stride=2,padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                               out_channels=3,
                                               kernel_size=4,
                                               stride=2,padding=1)
    def forward(self,inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = nn.functional.relu(x)
        return self._conv_trans_2(x)

class VQVAE(nn.Module):
    def __init__(self, num_hiddens=128, num_residual_layers=3, num_residual_hiddens=32, num_embeddings=512, 
    embedding_dim=512, commitment_cost=0.25, decay=0): # 원래는 num_embedding, embedding_dim = 512
        super(VQVAE, self).__init__()
        self._encoder_= Encoder(3,num_hiddens,num_residual_layers,num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                     out_channels=embedding_dim,
                                     kernel_size=1, # 원래는 1,1
                                     stride=1) # channel: 128 -> 512
        self._vq_vae = VectorQuantizer(num_embeddings,embedding_dim,commitment_cost)
            # dictionary 개수가 512개, 각 하나가 512차원
        self._decoder = Decoder(embedding_dim,
                              num_hiddens,
                              num_residual_layers,
                              num_residual_hiddens)
        self.resize = transforms.Resize(size = (270,480)) # 이거 없이는 B,C,268, 480 됨

    def forward(self, x):
        z = self._encoder_(x)
        z = self._pre_vq_conv(z)
        #print(f"before vqvae: {z.shape}")
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        x_recon = self.resize(x_recon)
        return loss, x_recon, perplexity

class VideoVQVAE(nn.Module):
    def __init__(self, in_channels=64, feature_dim=3072, latent_dim=256, dropout=0.0):
        super().__init__()
        self.conv_vqvae = VQVAE()
        self.encoder_rnn = nn.RNN(
            input_size = feature_dim, 
            hidden_size = feature_dim, #laten_dim
            num_layers = 1,
            nonlinearity = 'relu',
            batch_first = True
        )
        self.decoder_rnn = nn.RNN(
            input_size = feature_dim, #latent_dim
            hidden_size = feature_dim,
            num_layers = 1,
            nonlinearity = 'relu',
            batch_first = True
        )

    def encoding(self, x, h0=None):
        #self.conv_vqvae.load_state_dict(torch.load("../log/Jan23_16:47:06_<class '__main__.VQVAE'>/<class '__main__.VQVAE'>_best.pt"))
        #print(f"before view: {x.shape}")
        batch_size, seq_len = x.shape[:2]
        #x = x.view(-1, *x.shape[2:])

        # Encoder: CNN
        #print(f"at encoding: {x.shape}")
        features = self.conv_vqvae._encoder_(x)
        z = self.conv_vqvae._pre_vq_conv(features)
        #print(f"pre vqvae: {z.shape}")
        z = z.view(batch_size, seq_len, -1)
        #print(z.shape)

        # Encoder: RNN
        z, hn = self.encoder_rnn(z, h0)
        z = z.unsqueeze(-1)
        #print(f"after rnn: {z.shape}")

        #z = self.conv_vqvae._pre_vq_conv(features)

        return z, hn

    def decoding(self, x, h0=None):
        # Decoder: RNN
        #x = x.view(-1, *x.shape[2:])
        #print(f"in decoding: {x.shape}")
        batch_size, seq_len = x.shape[:2]
        x = x.reshape(batch_size, seq_len, -1)
        #print(f"in decoding squeeze : {x.shape}")
        features, hn = self.decoder_rnn(x, h0)
        #features = features.view(batch_size * seq_len, -1)
        #print(f"after RNN decoding: {features.shape}")

        #Decoder: CNN
        #features = features.view(256, 4, 9)
        features = features.reshape(batch_size, 256, 4, -1)
        #print(f"after view: {features.shape}")
        recon = self.conv_vqvae._decoder(features)
        recon = self.conv_vqvae.resize(recon)
        #print(f"at decoding - {recon.shape}")
        #recon = recon.unsqueeze(0)
        #print(f"after unsqeeze: {recon.shape}")
        return recon, hn

    def forward(self, x, enc_h0=None, dec_h0=None):
        #print(x.shape)
        z, enc_hn = self.encoding(x, h0=enc_h0)
        loss, quantized, perplexity, _ = self.conv_vqvae._vq_vae(z)
        #print(f"after vqvae: {quantized.shape}")
        recon, dec_hn = self.decoding(quantized, h0=dec_h0)
        #print(f"after decoding: {recon.shape}")
        #print(f"loss: {loss.shape}")
        return loss, recon, perplexity


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model

def show(img, type):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
    if type=='recon':
        plt.savefig('vqvae_test_recon.png')
    elif type=='ori':
        plt.savefig('vqvae_test_ori.png')

def train_one_epoch(model, dataloader, optimizer, criterion, scaler):
    model.train()

    losses = []
    for x in dataloader:
        x = x.to(device) #model.device
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            vq_loss, recon, perplexity = model(x)
            # loss = criterion(_____)
            recon_loss = nn.functional.mse_loss(recon, x) # / data_variance 
            loss = recon_loss + vq_loss
            #print(f"loss: {loss}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.detach().cpu().item())

    return np.mean(losses)

def eval(model, dataloader, criterion, scaler):
    model.eval()

    losses = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device) # model.device에서 오류나서 바꿈.
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                vq_loss, recon, perplexity = model(x)
                recon_loss = nn.functional.mse_loss(recon, x)
                loss = recon_loss + vq_loss
                losses.append(loss.detach().cpu().item())

            return np.mean(losses)

def recon_and_plot(model, recon_dataset, epoch, log_path=None): # dataset : example dataset
    def set_imshow_plot(ax):
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    model.eval()
    with torch.no_grad():
        _, preds, _ = model(recon_dataset.to(device))
        preds = preds.detach().cpu()
    print(f"***************{recon_dataset.shape}*******************")
    print(preds.shape)
    fig, axs = plt.subplots(3, 6, figsize=(18, 7), dpi=200)
    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.0)
    for i in range(3):
        for j in range(3):
            axs[i][j * 2].imshow(to_pil_image(recon_dataset[i * 3 + j, ...], mode='RGB'))
            axs[i][j * 2 + 1].imshow(to_pil_image(preds[i * 3 + j, ...], mode='RGB'))
            axs[i][j * 2].set_title("Image")
            axs[i][j * 2 + 1].set_title("Reconstructed")
            set_imshow_plot(axs[i][j * 2])
            set_imshow_plot(axs[i][j * 2 + 1])

    fig.suptitle(f"Epoch: {epoch}", y=0.99, fontsize=16)
    if log_path is not None:
        plt.savefig(os.path.join(log_path, 'recon', f"epoch_{epoch}.png"))

def plot_progress(train_losses, eval_losses, eval_step, log_path=None):
    _, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    train_x = np.arange(len(train_losses)) + 1
    eval_x = np.arange(eval_step, len(train_losses) + 1, eval_step)

    ax.set_title("Loss")
    ax.plot(train_x, train_losses, label="Train")
    ax.plot(eval_x, eval_losses, label="Eval")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    if log_path is not None:
        plt.savefig(os.path.join(log_path, f"loss.png"))


args = get_args()
args.model = VQVAE # MODEL_DICT 설정 아직 안해서
set_figure_options()
set_seed(seed=args.seed)
configure_cudnn(debug=args.debug)

device, num_workers = get_system_info()
print(f"Device: {device} | Seed: {args.seed} | Debig : {args.debug}")
print(args)
if args.eval_mode == 0:
    train_indices = None
    print("Split by even/odd-numbered")
else:
    train_indices = sorted(random.sample(range(30), 24))
    print("Stratified video split")
    print(f"Train data indices: {[x + 1 for x in train_indices]}")

# Dataset & Dataloader
transform = transform=transforms.Compose([
    transforms.Resize(size=tuple(args.img_size)),
    transforms.ToTensor(),
])

train_dataset = RandomFrameDataset(
    args.data_dir,
    transform=transform,
    train=True,
    train_indices=train_indices,
    eval_mode=args.eval_mode,
    debug=args.debug
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers
)

eval_dataset = RandomFrameDataset(
    args.data_dir,
    transform=transform,
    train=False,
    train_indices=train_indices,
    eval_mode=args.eval_mode,
    debug=args.debug,
)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=num_workers
)

recon_dataset = get_recon_dataset(args.data_dir, transform=transform)
#data_variance = np.var(train_dataset.data/ 255.0)

# Model, Criterion, Optimizer
model = VQVAE().to(device)
#model = load_checkpoint('../checkpoint.pth') # weight 불러오기
criterion = LOSS_DICT[args.loss]() # 이거 사실상 안씀
optimizer = OPTIM_DICT[args.optim](model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

# Logging
if args.log:
    log_path = f"../log/{datetime.today().strftime('%b%d_%H:%M:%S')}_{args.model}"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(log_path, 'recon'), exist_ok=True)
    with open(os.path.join(log_path, 'hps.txt'), 'w') as f: # Save h.p.s
        json.dump(args.__dict__, f, indent=4, default=str)
    with open(os.path.join(log_path, 'model.txt'), 'w') as f: # Save model structure
        f.write(model.__str__())

# Training & Evaluation
with torch.autograd.set_detect_anomaly(True):
    best_loss = float('inf')
    train_losses = []
    eval_losses = []
    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, scaler)
        train_losses.append(train_loss)
        if (epoch+1) % args.eval_step == 0:
            eval_loss = eval(model, eval_dataloader, criterion, scaler)
            eval_losses.append(eval_loss)
            if best_loss > eval_loss:
                best_loss = eval_loss
                if args.log:
                    torch.save(model.state_dict(), os.path.join(log_path, f"{args.model}_best.pt"))
                print(f"Best model at Epoch {epoch+1}/{args.epochs}, Best eval loss: {best_loss:.5f}")
            print(f"[Epoch {epoch + 1}/{args.epochs}] Train loss: {train_loss:.5f} | Eval loss: {eval_loss:.5f} | Best loss: {best_loss:.5f}")
            if args.log:
                recon_and_plot(model, recon_dataset, epoch+1, log_path)
                plot_progress(train_losses, eval_losses, args.eval_step, log_path)
                torch.save(model.state_dict(), os.path.join(log_path, f"{args.model}_last_epoch.pt"))

"""
## Take a random single batch
'''
for i in range(randrange(20)):
    for x in train_dataloader:
        print(x)
    (valid_originals, _) = next(iter(train_dataloader))
'''

for valid_originals in train_dataloader:
    model.eval()

    valid_originals = valid_originals.to(device)
    
    # 결국 아래 과정들이 model에 data 넣은건데.. 왜 따로 따로 한거지?
    vq_output_eval = model._pre_vq_conv(model._encoder_(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)

    '''
    shape = 5,512,8,8
    noise = Normal(0,1).sample(shape)
    valid_quantize = valid_quantize + noise
    '''
    valid_reconstructions = model._decoder(valid_quantize)
    

#     trans = transforms.ToPILImage()
#     for j in range(5):
#         valid_image = valid_reconstructions[i,:,:,:]
#         valid_image = trans(valid_image)
#         name = 'image_'+str(i)+str(j)+'.jpg'
#         valid_image.save(name, "JPEG")
    show(make_grid(valid_reconstructions.cpu().data), type='recon')
    show(make_grid(valid_originals.cpu().data), type='ori')
    break
"""
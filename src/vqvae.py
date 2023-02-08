from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
from utils import get_args, get_system_info, set_figure_options, set_seed, configure_cudnn
from dataset import RandomFrameDataset, get_recon_dataset
from loss import LOSS_DICT
from optimizer import OPTIM_DICT
from datetime import datetime
import os
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import json

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        ### Create an embedding matrix with size number of embedding X embedding dimension
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous() # (b,h,w,c), c=embedding_dim
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim) # (b*h*w, c)
        
        # Calculate distances between flattened input and embedding vector
        # distance btw pixels(72360) ~ vectors(512)
        # flat_input size: (b*h*w,512), embedding space size: (512,512)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t())) # a^2 + b^2 - 2ab
        # distance size : (72360, 512)
           
        # Choose indices that are min in each row
        # nearest vector for each pixels // need to return this!
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (72360, 1)

        ## Create a matrix of dimensions B*H*W into number of embeddings
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        ### Convert index to on hot encoding 
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        # embedding vector of given index 
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape) # (b,h,w,512)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
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
                     kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                     out_channels=num_hiddens,
                     kernel_size=1,stride=1,bias=False)
        )
        
    def forward(self,x):
        return x + self._block(x)
class ResidualStack(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(ResidualStack,self).__init__()
        self._num_residual_layers=num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels,num_hiddens,num_residual_hiddens) for _ in range(self._num_residual_layers)])
    def forward(self,x):
        for i in range(self._num_residual_layers):
            x=self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(Encoder,self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens//2,
                                kernel_size=4,
                                stride=2,padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels = num_hiddens,
                                 kernel_size=4,
                                 stride=2,padding=1
                                )
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1,padding=1)
        self._residual_stack = ResidualStack(in_channels = num_hiddens,
                                             num_hiddens = num_hiddens,
                                             num_residual_layers = num_residual_layers,
                                             num_residual_hiddens = num_residual_hiddens
                                            )
    def forward(self,inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        x = self._residual_stack(x)
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
        x = F.relu(x)
        return self._conv_trans_2(x)

class Model(nn.Module):
    def __init__(self,num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost,decay=0):
        super(Model,self).__init__()
        self._encoder_= Encoder(3,num_hiddens,num_residual_layers,num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                     out_channels=embedding_dim,
                                     kernel_size=1,
                                     stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings,embedding_dim,commitment_cost)
        self._decoder = Decoder(embedding_dim,
                              num_hiddens,
                              num_residual_layers,
                              num_residual_hiddens)
    def forward(self,x):
        z = self._encoder_(x)
        z = self._pre_vq_conv(z)
        loss,quantized,perplexity, latent = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity, latent # latent is one-hot encoded (nearest embedding vecotor for all pixels)
            
        
num_training_updates = 150
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 3
embedding_dim= 512
num_embeddings = 512
commitment_cost = 0.25
learning_rate = 3e-4

# import pre-trained weight
def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        #model.eval()
        return model


# get the dataset
args = get_args()
args.model = Model # MODEL_DICT 설정 아직 안해서
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

model = Model(num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost,decay=0).to(device)
model = load_checkpoint('../checkpoint.pth')
criterion = LOSS_DICT[args.loss]() # 이거 사실상 안씀
optimizer = OPTIM_DICT[args.optim](model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

def train_one_epoch(model, dataloader, optimizer, criterion, scaler):
    model.train()

    losses = []
    for x in dataloader:
        #x = x.to(device) #model.device
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            vq_loss, quantized, recon, perplexity = model(x)
            # loss = criterion(_____)
            recon_loss = nn.functional.mse_loss(recon, x) # / data_variance 
            loss = recon_loss + vq_loss
            #print(f"loss: {loss}")

        optimizer.zero_grad()
        loss.requires_grad_(True)
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
            #x = x.to(device) # model.device에서 오류나서 바꿈.
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                vq_loss, quantized, recon, perplexity = model(x)
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
        #_, preds, _ = model(recon_dataset.to(device))
        _, quantized, preds, _ = model(recon_dataset)
        print(f"output quantized : {quantized.shape}")
        preds = preds.detach().cpu()
        
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
is_train=False

with torch.autograd.set_detect_anomaly(True):
    best_loss = float('inf')
    train_losses = []
    eval_losses = []

    if is_train:
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
    else:
        #eval_loss = eval(model, eval_dataloader, criterion, scaler)
        #eval_losses.append(eval_loss)
        if args.log:
            recon_and_plot(model, recon_dataset, 0, log_path)

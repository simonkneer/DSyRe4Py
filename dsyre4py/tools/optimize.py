
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import random_split
from torch import linalg as LA
import pytorch_lightning as pl
import numpy as np

size = 0


class h5dataset(Dataset):
    def __init__(self,data):
        data_re = np.real(data)
        data_im = np.imag(data)
        data = np.stack([data_re,data_im],axis=1)
        self.data = np.stack([data[:-1],data[1:]],axis=1)

    def __getitem__(self,index):

        return([self.data[index,0,0,:],self.data[index,0,1,:],self.data[index,1,0,:],self.data[index,1,1,:]])
    def __len__(self):
        return self.data.shape[0]


class lit_optim(pl.LightningModule):
    def __init__(self,lr=1e-2,size=10):
        super().__init__()
        self.lr = lr
                        
        self.optim_re = nn.Linear(size, 1, bias=False)
        self.optim_im = nn.Linear(size, 1, bias=False)


    def forward(self, r):
        x_re,x_im,y_re,y_im = r
        c_11_re = self.optim_re(x_re)
        c_11_im = self.optim_im(x_re)

        c_12_re = -self.optim_im(x_im)
        c_12_im = self.optim_re(x_im) 

        c_21_re = self.optim_re(y_re)
        c_21_im = self.optim_im(y_re)

        c_22_re = -self.optim_im(y_im)
        c_22_im = self.optim_re(y_im) 

        return [torch.atan2(c_11_im+c_12_im,c_11_re+c_12_re),torch.atan2(c_21_im+c_22_im,c_21_re+c_22_re)]
    
    def loss_f(self,x,y):
        loss_1 = torch.sin((x-y)/2)
        return(torch.max((loss_1) ** 2))
    
    def loss_and_reg(self,x,y):
        loss = self.loss_f(x,y)
        regularization = (1 - LA.matrix_norm(self.optim_re.state_dict()['weight'])- LA.matrix_norm(self.optim_im.state_dict()['weight']))**2
        return loss + regularization
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = {
        'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.998),
        'interval': 'epoch'  # called after each training step            
        }
        return optimizer

    def training_step(self, train_batch, batch_idx):
        phase_1,phase_2 = self.forward(train_batch)    #this can be just self(x) whatever
        loss = self.loss_and_reg(phase_1, phase_2)
        self.log('train_loss', self.loss_f(phase_1, phase_2), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


class lit_optim_real(pl.LightningModule):
    def __init__(self,lr=1e-2,size=10):
        super().__init__()
        self.lr = lr
                        
        self.optim_re = nn.Linear(size, 1, bias=False)


    def forward(self, r):
        x_re,x_im,y_re,y_im = r
        c_11_re = self.optim_re(x_re)
        c_12_im = self.optim_re(x_im) 
        c_21_re = self.optim_re(y_re)
        c_22_im = self.optim_re(y_im) 

        return [torch.atan2(c_12_im,c_11_re),torch.atan2(c_22_im,c_21_re)]
    
    def loss_f(self,x,y):
        loss_1 = torch.sin((x-y)/2)
        return(torch.max((loss_1) ** 2))
    
    def loss_and_reg(self,x,y):
        loss = self.loss_f(x,y)
        regularization = (1 - LA.matrix_norm(self.optim_re.state_dict()['weight']))**2
        return loss + regularization
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = {
        'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.998),
        'interval': 'epoch'  # called after each training step            
        }
        return optimizer

    def training_step(self, train_batch, batch_idx):
        phase_1,phase_2 = self.forward(train_batch)    #this can be just self(x) whatever
        loss = self.loss_and_reg(phase_1, phase_2)
        self.log('train_loss', self.loss_f(phase_1, phase_2), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


def optimize(data,*,learning_rate=1E-2,epochs=500,batch_size=200,ax=None,complex_val=True):
    """This is a simple routine that optimizes weights to minimize the temporal derivate of the symmetry reducing phase.


    Args:
        data (array): Data that is used to generate symmetry reducing phase
        learning_rate (float, optional): Learning rate. Defaults to 1E-2.
        epochs (int, optional): training epochs. Defaults to 500.
        batch_size (int, optional): Batch size. Defaults to 200.
        ax (matplotlib axes, optional): Axes for plotting the phase. Defaults to None.
        complex_val (bool, optional): Whether the weights are real or complex valued. Defaults to True.

    Returns:
        array: optimized weights
    """
    dataset = h5dataset(data)

    train_loader = DataLoader(dataset, batch_size=batch_size)
    if complex_val:
        model = lit_optim(learning_rate,size=data.shape[1])
    else:
        model = lit_optim_real(learning_rate,size=data.shape[1])

    trainer = pl.Trainer(accelerator="cpu",devices=1, precision=64,max_epochs=epochs,enable_checkpointing=False)
    trainer.fit(model, train_loader)#, val_loader)

    weight_re = model.optim_re.state_dict()['weight'].detach().numpy()
    if complex_val:
        weight_im = model.optim_im.state_dict()['weight'].detach().numpy()
    if ax != None:
        plot_loader = DataLoader(dataset, batch_size=data.shape[0])
        outs = trainer.predict(model,plot_loader)
        print(len(outs),len(outs[0]),data.shape)
        print(outs[0][0].detach().numpy().shape)
        ax.plot(np.squeeze(outs[0][0].detach().numpy()))
    if complex_val:        
        return [np.squeeze(weight_re),np.squeeze(weight_im)]
    else:
        return np.squeeze(weight_re)
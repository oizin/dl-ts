# bayesian_lstm_dropout.py
# Based on Zhu and Laptev (2017) paper

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BayesLSTMencoder(nn.Module):
    """
    A Bayesian LSTM with a pretrained time series encoder
    """
    def __init__(self, encoder,n_features,hidden_dim,output_dim):
        super(BayesLSTMencoder, self).__init__()
        self.encoder = encoder
        self.encoder_dim = encoder.hidden_size
        self.hidden_size = hidden_dim
        self.l1 = nn.Linear(encoder.hidden_size+n_features,hidden_dim)
        self.l2 = nn.Linear(hidden_dim,output_dim)
        
    def forward(self, x_encoder, x_other,dropout=False):
        _,context = self.encoder(x_encoder) 
        context = F.dropout(context[0], p=0.5, training=dropout)
        output = torch.cat((context.view(-1,1,self.encoder_dim), x_other), dim=2)
        output = self.l1(F.relu(output))
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.l2(F.relu(output))
        return output
    
    def train_single_epoch(self, dataloader, optimizer, loss_fn):
        """
        train Bayesian LTSM with encoder

        : param dataloader:              dataloader
        : optimizer
        : loss_fn
        : return losses:                   array of loss function for each epoch
        """
        batch_size = dataloader.batch_size
        epoch_loss = 0.

        for i, (x_encoder,x_other,target_tensor) in enumerate(dataloader):

            # zero the gradient
            optimizer.zero_grad()

            # predict
            outputs = self.forward(x_encoder, x_other, True)

            # compute the loss 
            loss = loss_fn(outputs, target_tensor)
            epoch_loss += loss.item()

            # backpropagation
            loss.backward()
            optimizer.step()

        # loss for epoch 
        epoch_loss /= len(dataloader) 

        return epoch_loss

    def inference(self,x_encoder, x_other,dropout=False):
        """
        Inference on newly observed data
        """
        # encode input_tensor
        x_encoder = x_encoder.unsqueeze(0)
        x_other = x_other.unsqueeze(0)

        # predict
        outputs = self.forward(x_encoder,x_other,dropout)
        np_outputs = outputs.detach().cpu().numpy()
        
        return np_outputs

    
# class BayesianLSTM(nn.Module):
#     """
#     A Bayesian LSTM
    
#     adapted from: https://github.com/PawaritL/BayesianLSTM
#     """
#     def __init__(self, n_features, output_length):

#         super(BayesianLSTM, self).__init__()

#         self.hidden_size_1 = 128
#         self.hidden_size_2 = 32
#         self.n_layers = 1 # number of (stacked) LSTM layers for each stage

#         self.lstm1 = nn.LSTM(n_features, 
#                              self.hidden_size_1, 
#                              num_layers=1,
#                              batch_first=True)
#         self.lstm2 = nn.LSTM(self.hidden_size_1,
#                              self.hidden_size_2,
#                              num_layers=1,
#                              batch_first=True)
        
#         self.dense = nn.Linear(self.hidden_size_2, output_length)
#         self.loss_fn = nn.MSELoss()
        
#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()

#         hidden = self.init_hidden1(batch_size)
#         output, _ = self.lstm1(x, hidden)
#         output = F.dropout(output, p=0.5, training=True)
#         state = self.init_hidden2(batch_size)
#         output, state = self.lstm2(output, state)
#         output = F.dropout(output, p=0.5, training=True)
#         output = self.dense(state[0].squeeze(0))
        
#         return output
        
#     def init_hidden1(self, batch_size):
#         hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
#         cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
#         return hidden_state, cell_state
    
#     def init_hidden2(self, batch_size):
#         hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_2))
#         cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_2))
#         return hidden_state, cell_state
    
#     def loss(self, pred, truth):
#         return self.loss_fn(pred, truth)

#     def predict(self, X):
#         return self.forward(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()


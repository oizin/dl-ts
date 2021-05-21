# lstm_encoder_decoder.py
# adapted from: https://github.com/lkulowski/LSTM_encoder_decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers = 1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        '''
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)

    def forward(self, x_input):
        '''
        : param x_input:               input of shape (# in batch, seq_len, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        lstm_out, self.hidden = self.lstm(x_input)
        
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        return (torch.zeros(batch_size, self.num_layers, self.hidden_size),
                torch.zeros(batch_size, self.num_layers, self.hidden_size))
    
class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    def __init__(self, input_size, hidden_size, num_layers = 1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
        '''
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(1), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden
    
class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, input_size, hidden_size):
        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''
        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)

    def train_single_epoch(self, dataloader, optimizer, loss_fn):
        '''
        train lstm encoder-decoder
        
        : param dataloader:              dataloader   
        : optimizer
        : loss_fn
        : return losses:                   array of loss function for each epoch
        '''
        batch_size = dataloader.batch_size
        epoch_loss = 0.

        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            
            # initialize hidden state
            encoder_hidden = self.encoder.init_hidden(batch_size)

            # zero the gradient
            optimizer.zero_grad()

            # predict
            target_len = target_tensor.size(1)
            outputs = self.forward(input_tensor,target_len)
            
            # compute the loss 
            loss = loss_fn(outputs, target_tensor)
            epoch_loss += loss.item()

            # backpropagation
            loss.backward()
            optimizer.step()

        # loss for epoch 
        epoch_loss /= len(dataloader) 
                    
        return epoch_loss
    
    def evaluate(self,dataloader, loss_fn):
        
        batch_size = dataloader.batch_size
        eval_loss = 0.

        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            # initialize hidden state
            encoder_hidden = self.encoder.init_hidden(batch_size)
            # predict
            target_len = target_tensor.size(1)
            outputs = self.forward(input_tensor,target_len)
            # compute the loss 
            loss = loss_fn(outputs, target_tensor)
            eval_loss += loss.item()

        # loss for evaluation 
        eval_loss /= len(dataloader) 
                    
        return eval_loss
    
    def forward(self, input_tensor, target_len):
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''
        # encode input_tensor
        input_tensor = input_tensor#.unsqueeze(0)     # add in batch size of 1
        batch_size = input_tensor.size(0)
        input_size = input_tensor.size(1)
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(batch_size, target_len, input_tensor.size(2))

        # encoder -> decoder
        decoder_input = input_tensor[:,input_size-1,:]   # shape: (batch_size, input_size)
        decoder_hidden = encoder_hidden

        # predict
        for t in range(target_len): 
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:,t,:] = decoder_output.squeeze(1)
            decoder_input = input_tensor[:,input_size-t-1,:]
                    
        return outputs

    def inference(self,input_tensor,target_len):
        """
        Inference on newly observed data
        """
        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(0)     # add in batch size of 1
        batch_size = input_tensor.size(0)
        input_size = input_tensor.size(1)
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(batch_size, target_len, input_tensor.size(2))

        # encoder -> decoder
        decoder_input = input_tensor[:,input_size-1,:]   # shape: (batch_size, input_size)
        decoder_hidden = encoder_hidden

        # predict
        for t in range(target_len): 
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:,t,:] = decoder_output.squeeze(1)
            decoder_input = input_tensor[:,input_size-t-1,:]
                    
        np_outputs = outputs.detach().cpu().numpy()
        
        return np_outputs

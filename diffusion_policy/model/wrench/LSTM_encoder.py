import torch
import einops
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self,  input_dim, hidden_dim, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        # self.dropout = dropout

        self.lstm_enc = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                                dropout=0, batch_first=True,num_layers=num_layers)

    def forward(self, x):
        B,L,H = x.shape
        out, (last_h_state, last_c_state) = self.lstm_enc(x)
        # x_enc = last_h_state.squeeze(dim=0)
        x_enc =  einops.rearrange(last_h_state,  'n b h -> b n h') # n is the number of layers
        x_enc = x_enc.reshape(B,1,-1)
        x_enc = x_enc.repeat(1, x.shape[1], 1)
        last_h_state = last_h_state.reshape(B,-1)

        return last_h_state

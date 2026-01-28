import torch.nn as nn

# encoder class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, seq_len):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_len = seq_len

        self.lstm_enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

    def forward(self, x):
        out, (last_h_state, last_c_state) = self.lstm_enc(x)
        x_enc = last_h_state.squeeze(dim=0)
        x_enc = x_enc.unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_enc, out
    
# decoder class 
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, seq_len, use_act):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_len = seq_len
        self.use_act = use_act
        self.act = nn.Sigmoid

        self.lstm_dec = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, z): 
        dec_out, (hidden_state, cell_state) = self.lstm_dec(z)
        dec_out = self.fc(dec_out)
        if self.use_act: 
            dec_out = self.act(dec_out)
        return dec_out, hidden_state
    
# lstm ae class 
class LSTMAE(nn.Module): 
    def __init__(self, input_size, hidden_size, dropout_ratio, seq_len, use_act=True):
        super(LSTMAE).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.seq_len = seq_len

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout_ratio, seq_len=seq_len)
        self.decoder = Decoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout_ratio, seq_len=seq_len, use_act=use_act)

    def forward(self, x, return_last_h=False, return_enc_out=False): 
        x_enc, enc_out = self.encoder(x)
        x_dec, last_h = self.decoder(x_enc)

        if return_last_h: 
            return x_dec, last_h
        elif return_enc_out: 
            return x_dec, enc_out
        return x_dec
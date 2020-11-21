import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 3000

class LSTMModel(nn.Module):
    def __init__(self):
        """Construct LSTM model.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=161, hidden_size=1024, num_layers=2, batch_first=True, dropout=0.4)

    def forward(self, ipt):
        self.lstm.flatten_parameters()
        o, h = self.lstm(ipt)
        return o , h

class EncoderRNN(nn.Module):
    def __init__(self):       # hidden_size = 1024   input_size=161
        super().__init__()
        self.lstm = nn.LSTM(input_size=161, hidden_size=1024, num_layers=2, batch_first=True, dropout=0.4)

    def forward(self, inp):
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(inp)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inp, hidden, encoder_outputs):
        # embedded = self.embedding(input).view(1, 1, -1) 
        # embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((inp[0], hidden[0]), 1)), dim=1)


        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((inp[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)

        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)                

if __name__ == '__main__':
    hidden_size = 1024
    ipt = torch.rand(1, 355, 161)

    print(ipt.device)
    # opt = LSTMModel()(ipt)
    
    encoder1 = EncoderRNN()
    decoder = AttnDecoderRNN(hidden_size, 161, dropout_p=0.1)
    o , h = encoder1(ipt)

    a,b,c = decoder(ipt,h,o)
    print(a)
    print(b)
    print(c)
    # print(opt.shape)
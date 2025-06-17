import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Definition (from SG6z_vqJzICK)
class Encoder(nn.Module):
    def __init__(self, inp_dim, emb_dim, hid_dim, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(inp_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=0.3) # Using hardcoded dropout from cell
        self.fc = nn.Linear(hid_dim*2, hid_dim)
        self.dropout = nn.Dropout(0.3) # Using hardcoded dropout from cell
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim + hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim, hid_dim, attention, n_layers=2):
        super().__init__()
        self.output_dim = out_dim
        self.attention = attention
        self.embedding = nn.Embedding(out_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(hid_dim * 2 + emb_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=0.3) # Using hardcoded dropout
        self.fc_out = nn.Linear(hid_dim*3+emb_dim, out_dim)
        self.dropout = nn.Dropout(0.3) # Using hardcoded dropout
        self.n_layers = n_layers

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        embedded, output, weighted = embedded.squeeze(1), output.squeeze(1), weighted.squeeze(1)
        pred_input = torch.cat((output, weighted, embedded), dim=1)
        prediction = self.fc_out(pred_input)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder; self.decoder = decoder; self.device = device
        self.n_layers = decoder.n_layers

    def forward(self, src, trg, teacher_forcing_ratio=0.75):
        batch, trg_len = trg.shape
        outputs = torch.zeros(batch, trg_len, self.decoder.output_dim).to(self.device)
        encoder_outputs, encoder_hidden = self.encoder(src)
        hidden = torch.tanh(self.encoder.fc(torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=1))).unsqueeze(0)
        hidden = hidden.repeat(self.n_layers, 1, 1)
        input = trg[:,0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:,t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:,t] if teacher_force else top1
        return outputs


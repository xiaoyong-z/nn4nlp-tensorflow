import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
criterion = torch.nn.CrossEntropyLoss()
type = torch.LongTensor

if use_cuda:
    type = torch.cuda.LongTensor
class encoder(torch.nn.Module):
    def __init__(self, nwords_trg, emb_size, hidden_dim, num_layer, use_cuda):
        super(encoder, self).__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.embed = torch.nn.Embedding(nwords_trg, emb_size)
        torch.nn.init.uniform_(self.embed.weight, -0.25, 0.25)
        self.lstm = torch.nn.LSTM(input_size=emb_size, hidden_size=hidden_dim, num_layers=num_layer,
                                     batch_first=True)


    def forward(self, source, src_length=None, hidden=None):
        src_emb = self.embed(source)
        src_emb = src_emb.unsqueeze(0)
        if hidden is None:
            if self.use_cuda:
                h0 = Variable(torch.zeros(self.num_layer, 1, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layer, 1, self.hidden_dim).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layer, 1, self.hidden_dim))
                c0 = Variable(torch.zeros(self.num_layer, 1, self.hidden_dim))
            hidden = (h0, c0)

        # if src_length is not None:
        #     src_emb = torch.nn.utils.rnn.pack_padded_sequence(src_emb, src_length, batch_first=True)

        output, enc_h_t = self.lstm(src_emb, hidden)

        # if src_length is not None:
        #     output, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_h, batch_first=True)

        return enc_h_t


class decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.dec2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, trg, hidden):
        '''
        enc_h  : B x S x H
        prev_s : B x H
        '''
        src_emb = self.embed(trg)
        src_emb = src_emb.unsqueeze(0).unsqueeze(1)
        output, hidden = self.lstm(src_emb, hidden)
        output = self.dec2word(output).squeeze(1)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim, hidden_dim, max_len, trg_soi, trg_eos, use_cuda):
        super(Seq2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword
        self.max = max_len
        self.trg_soi = trg_soi
        self.trg_eos = trg_eos

        self.encoder = encoder(src_nword, embed_dim, hidden_dim, num_layer, use_cuda)
        self.decoder = decoder(trg_nword, embed_dim, hidden_dim)

    def forward(self, source, target=None):
        total_loss = []
        trg_sent = []
        if target is not None:
            enc_h = self.encoder(source)
            hidden = enc_h

            for i in range(target.shape[0] - 1):
                output, hidden = self.decoder(target[i], hidden)
                loss = criterion(output, target[i+1].unsqueeze(0))
                total_loss.append(loss)
        else:
            enc_h = self.encoder(source)
            hidden = enc_h

            trg_word = torch.tensor(self.trg_soi).type(type)
            i = 0
            trg_sent.append(self.trg_soi)
            for i in range(self.max):
                output, hidden = self.decoder(trg_word, hidden)
                # if torch.argmax(output, dim=1)[0].item() == self.trg_soi:
                #     trg_word = torch.sort(output, dim=1)[1][0][-2]
                # else:
                trg_word = torch.argmax(output, dim=1)[0]
                trg_sent.append(trg_word.item())
                if(trg_word == self.trg_eos):
                    break
        return total_loss, trg_sent

import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
criterion = torch.nn.CrossEntropyLoss(reduce="none")
type = torch.LongTensor

if use_cuda:
    type = torch.cuda.LongTensor


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.enc_h_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.prev_s_in = nn.Linear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, enc_h, prev_s):
        '''
        enc_h  : B x S x 2*H
        prev_s : B x 1 x H
        '''
        seq_len = enc_h.size(1)

        enc_h_in = self.enc_h_in(enc_h)  # B x S x H
        prev_s = self.prev_s_in(prev_s).unsqueeze(1)  # B x 1 x H

        h = F.tanh(enc_h_in + prev_s.expand_as(enc_h_in))  # B x S x H
        h = self.linear(h)  # B x S x 1

        alpha = F.softmax(h)
        ctx = torch.bmm(alpha.transpose(2, 1), enc_h).squeeze(1)  # B x 1 x 2*H

        return ctx

class encoder(torch.nn.Module):
    def __init__(self, nwords_trg, emb_size, hidden_dim, num_layer, use_cuda):
        super(encoder, self).__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.embed = torch.nn.Embedding(nwords_trg, emb_size)
        torch.nn.init.uniform_(self.embed.weight, -0.25, 0.25)
        self.lstm = torch.nn.LSTM(input_size=emb_size, hidden_size=hidden_dim, num_layers=num_layer,
                                     batch_first=True, bidirectional=True)


    def forward(self, source, src_length=None, hidden=None):
        batch_size = source.size(0)
        src_emb = self.embed(source)
        if hidden is None:
            if self.use_cuda:
                h0 = Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_dim).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_dim))
                c0 = Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_dim))
            hidden = (h0, c0)

        output, hidden = self.lstm(src_emb, hidden)

        return output, hidden


class decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_len, trg_soi):
        super(decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.trg_soi = trg_soi

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_dim)
        self.decodercell = DecoderCell(embed_dim, hidden_dim)
        self.dec2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_h, prev_s, target, train=True):
        '''
        enc_h  : B x S x 2*H
        prev_s : B x H
        '''

        if train:
            batch_size, target_len = target.size(0), target.size(1)

            dec_h = Variable(torch.zeros(batch_size, target_len, self.hidden_dim))

            if torch.cuda.is_available():
                dec_h = dec_h.cuda()

            target = self.embed(target)
            for i in range(target_len):
                ctx = self.attention(enc_h, prev_s)
                prev_s = self.decodercell(target[:, i], prev_s, ctx)
                dec_h[:, i, :] = prev_s

            outputs = self.dec2word(dec_h)
        else:
            target = self.embed(target).unsqueeze(0)
            ctx = self.attention(enc_h, prev_s)
            prev_s = self.decodercell(target, prev_s, ctx)
            outputs = self.dec2word(prev_s)
        return outputs, prev_s

class DecoderCell(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(DecoderCell, self).__init__()

        self.input_weights = nn.Linear(embed_dim, hidden_dim*2)
        self.hidden_weights = nn.Linear(hidden_dim, hidden_dim*2)
        self.ctx_weights = nn.Linear(hidden_dim*2, hidden_dim*2)

        self.input_in = nn.Linear(embed_dim, hidden_dim)
        self.hidden_in = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_in = nn.Linear(hidden_dim*2, hidden_dim)


    def forward(self, trg_word, prev_s, ctx):
        '''
        trg_word : B x E
        prev_s   : B x H
        ctx      : B x 2*H
        '''
        gates = self.input_weights(trg_word) + self.hidden_weights(prev_s) + self.ctx_weights(ctx)
        reset_gate, update_gate = gates.chunk(2,1)

        reset_gate = F.sigmoid(reset_gate)

        prev_s_tilde = self.input_in(trg_word) + self.hidden_in(prev_s) + self.ctx_in(ctx)
        prev_s_tilde = F.tanh(prev_s_tilde)

        prev_s = torch.mul((1-reset_gate), prev_s) + torch.mul(reset_gate, prev_s_tilde)
        return prev_s


class Seq2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim, attention_size, hidden_dim, max_len, trg_soi, trg_eos, use_cuda):
        super(Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword
        self.max = max_len
        self.trg_soi = trg_soi
        self.trg_eos = trg_eos
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.encoder = encoder(src_nword, embed_dim, hidden_dim, num_layer, use_cuda)
        self.decoder = decoder(trg_nword, embed_dim, hidden_dim, max_len, trg_soi)

    def forward(self, source, target=None, masks=None):
        batch_size = source.size(0)
        loss = []
        trg_sent = []
        if target is not None:
            output, hidden = self.encoder(source)
            prev_s = hidden[-1][-1] # B x H
            prev_s = F.tanh(self.linear(prev_s))  # B x H

            out, _ = self.decoder(output, prev_s, target)
            for i in range(target.size(1) - 1):
                loss.append(masks[:, i].float() * criterion(out[:,i,:], target[:,i + 1]))
            loss = torch.stack(loss).sum()
        else:
            output_prev, hidden = self.encoder(source.unsqueeze(0))
            prev_s = hidden[-1][-1]  # B x H
            prev_s = F.tanh(self.linear(prev_s))  # B x H
            trg_word = self.trg_soi
            trg_word = torch.tensor(trg_word).type(type)
            for i in range(self.max):
                output, prev_s =  self.decoder(output_prev, prev_s, trg_word, False)
                trg_word = torch.argmax(output[0], dim=0)
                trg_sent.append(trg_word.item())
                if (trg_word == self.trg_eos):
                    break
        return loss, trg_sent

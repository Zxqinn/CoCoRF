import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def embed ():
    # 预训练的 GloVe 词向量文件
    glove_file = './glove_DATA/python_glove.txt'
    weights = []
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            vector = list(map(float, line[1:]))
            weights.append(vector)
    weights = torch.tensor(weights)
    return weights


# Encoder
# ------------------------------------------------------------------------------

# Encode into Z with mu and log_var

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,  dropout_p=0.5, cuda=True):
        super(Encoder, self).__init__()
        self.cuda = cuda
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size)
        # 使用预训练词向量
        #self.embed = nn.Embedding.from_pretrained(embed())
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=1,dropout=dropout_p)
        self.o2p = nn.Linear(hidden_size, output_size * 2)

    def forward(self, input):
        embedded = self.embed(input)
        output, hidden = self.self_attention(embedded, embedded, embedded)
        output = output.mean(dim=1)
        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar)
        return mu, logvar, z

    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        if self.cuda:
            eps = eps.cuda()
        std = torch.exp(logvar / 2.0)
        return mu + eps * std


# Decoder
# ------------------------------------------------------------------------------

# Decode from Z into sequence


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.5, cuda=True):
        super(Decoder, self).__init__()
        self.SOS = 1
        self.cuda = cuda
        self.embed = nn.Embedding(output_size, input_size)
        # self.embed = nn.Embedding(output_size, hidden_size)
        # 使用预训练词向量
        # self.embed = nn.Embedding.from_pretrained(embed())
        self.output_size = output_size
        self.n_layers = n_layers

        self.self_attention = nn.MultiheadAttention(input_size, num_heads=1,dropout=dropout_p)

        self.output_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size)
        )
        self.generator = nn.Linear(input_size, output_size)

    def forward(self, hidden, trg):
        max_len = trg.size(1)
        hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)
        logit = []
        input = Variable(torch.LongTensor([self.SOS] * trg.size(0)))
        if self.cuda:
            input = input.cuda()
        for i in range(max_len):
            hidden, input = self.step(hidden, input)
            logit.append(input)
            use_teacher_forcing = random.random() <= 0.9 if self.training and i > 0 else 1
            if use_teacher_forcing:
                input = trg[:, i]
            else:
                input = input.argmax(dim=1)
        logit = torch.stack(logit, dim=1)
        return logit

    # def sample(self, hidden, trg):
    #     max_len = trg.size(1)
    #     hidden = hidden.unsqueeze(0).repeat(self.n_layers,1,1)
    #     logit = []
    #     for i in range(max_len):
    #         hidden, token_logit = self.step(hidden, input)
    #         input = token_logit.argmax(dim=1)
    #         logit.append(token_logit)
    #     logit = torch.stack(logit, dim=1)
    #     return logit

    def step(self, hidden, token):
        token_embedding = self.embed(token.unsqueeze(0))
        hidden, attn_output_weights = self.self_attention(token_embedding,hidden,hidden)
        output = self.output_projection(hidden.squeeze(0))
        token_logit = self.generator(output)
        return hidden, token_logit


# Container
# ------------------------------------------------------------------------------


class VAE(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_size, dropout=0, cuda=True):
        super(VAE, self).__init__()
        self.encoder = Encoder(vocab_size, hidden_size, emb_size, dropout_p=dropout, cuda=cuda)
        self.decoder = Decoder(emb_size, hidden_size, vocab_size, dropout_p=dropout, cuda=cuda)

    def forward(self, inputs):
        m, l, z = self.encoder(inputs)
        decoded = self.decoder(z, inputs)
        return m, l, z, decoded

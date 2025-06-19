import torch
import torch.nn as nn 

import copy
import math
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.functional import normalize


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    

def setEmbedingModel(d_list,d_out):
    return nn.ModuleList([nn.Linear(d,d_out) for d in d_list])


def setReEmbedingModel(d_list,d_out):
    return nn.ModuleList([nn.Linear(d_out,d)for d in d_list])


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) #scores shape is [bs heads view view]
    if mask is not None:
        mask = mask.unsqueeze(1).float()
        mask = mask.unsqueeze(-1).matmul(mask.unsqueeze(-2))#mask shape is [bs 1 view view]
        # mask = mask.unsqueeze(1) #mask shape is [bs 1 1 view]
        scores = scores.masked_fill(mask == 0, -1e9) # mask invalid view
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):

        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model/h
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.2):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        x = self.dropout_2(self.linear_2(x))
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)

        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
    

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        print(x)
        return self.embed(x)

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N

        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = src
        # x = self.embed(src)
        # x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N

        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder( d_model, N, heads, dropout)
    def forward(self, src, src_mask):
        e_outputs = self.encoder(src, src_mask)
        return e_outputs


class Model(nn.Module):
    def __init__(self, input_len, d_model, n_layers, heads, d_list, high_feature_dim, classes_num, dropout,recover=True):
        super().__init__()
        self.ETrans = Transformer(d_model, n_layers, heads, dropout)
        self.DTrans = Transformer(d_model, n_layers, heads, dropout)
        self.embeddinglayers = setEmbedingModel(d_list,d_model)
        self.re_embeddinglayers = setReEmbedingModel(d_list,d_model)
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(d_model, high_feature_dim),
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.recover = recover
        
    def forward_refactor(self,x,mask=None,recover = None):
        view_num = len(x)
        for i in range(view_num): # encode input view to features with same dimension 
            x[i] = self.embeddinglayers[i](x[i])
        z = [view.clone() for view in x]
        x = torch.stack(x,dim=1) # B,view,d
        x = self.ETrans(x,mask) if self.recover else self.ETrans(x,None)
        encX = x
        
        if recover: # Stage 1
            x = x.mul(mask.unsqueeze(2))
            x = torch.einsum('bvd->bd',x)
            wei = 1 / torch.sum(mask, 1)
            x = torch.diag(wei).mm(x)
        else: # Stage 2
            x = torch.einsum('bvd->bd',x)/view_num
        h = x.detach().clone()
        x = x.unsqueeze(1).expand(-1,view_num,-1)
        x = self.DTrans(x,None)
        decX = x 

        x_bar = [None]*view_num
        for i in range(view_num):       
            x_bar[i] = self.re_embeddinglayers[i](x[:,i])
        
        return encX,decX,x_bar,h,z
    
    def forward_contrast(self,xs,mask=None):
        
        x = [view.clone() for view in xs]  # 创建副本
        hs = []
        xrs = []
        view_num = len(x)
        for i in range(view_num):
            h = normalize(self.feature_contrastive_module(x[i]), dim=1)
            # xr = self.re_embeddinglayers[i](x[i])
            hs.append(h)
            #这个xrs有问题
            # xrs.append(xr)

        # encX_flat = encX.view(-1, encX.size(-1))  # Shape: (batch_size * view_num, d_model)
        # h = self.feature_contrastive_module(encX_flat)  # Shape: (batch_size * view_num, high_feature_dim)

        # # 归一化后恢复原始形状
        # h = normalize(h, dim=1)  # Normalize along the feature dimension
        # h = h.view(encX.size(0), encX.size(1), -1)  # Shape: (batch_size, view_num, high_feature_dim)
        
        return hs,xrs


def get_model(d_list,d_model=768,n_layers=2,heads=4,high_feature_dim=128,classes_num=10,dropout=0.2,load_weights=None,device=torch.device('cuda:0')):
    
    assert d_model % heads == 0
    assert dropout < 1

    model = Model(len(d_list), d_model, n_layers, heads, d_list, high_feature_dim, classes_num, dropout)

    if load_weights is not None:
        print("loading pretrained weights...")
        # model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    
    model = model.to(device)
    
    return model




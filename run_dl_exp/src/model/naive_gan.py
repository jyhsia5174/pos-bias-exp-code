import torch
import numpy as np
import torch.nn.functional as F

class Generator(torch.nn.Module):
    def __init__(self, inputSize, embed_dim):
        super().__init__()
        self.inputSize = inputSize
        self.embed1 = torch.nn.Embedding(inputSize, embed_dim, padding_idx=0)  
        torch.nn.init.xavier_uniform_(self.embed1.weight.data[1:, :])
        #torch.nn.init.uniform_(self.embed1.weight.data[1:, :], 0, 1./np.sqrt(self.inputSize*1.))


    def forward(self, ctx, itm, pos, ctx_v):  # ctx: context, itm: item, x3: position, ctx_v: context value
        if not self.training:  # ignore one-hot feature never seen in training phase
            ctx[ctx>=self.inputSize] = 0
            ctx_v[ctx>=self.inputSize] = 0
        ctx = torch.sum(torch.mul(self.embed1(ctx), ctx_v.unsqueeze(2)), dim=1)  # field 1 embedding for cxt: (batch_size, cxt_nonzero_feature_num, embed_dim)
        itm = torch.sum(self.embed1(itm), dim=1)  # field 1 embedding for item: (batch_size, item_nonzero_feature_num, embed_dim)

        ## merge
        x12 = torch.sum(ctx*itm, dim=1)  # (batch_size,)
        #print('G size', x12.size())
        return x12

class Generator_Z(torch.nn.Module):
    def __init__(self, inputSize, embed_dim):
        super().__init__()
        self.inputSize = inputSize
        self.embed1 = torch.nn.Embedding(inputSize, embed_dim, padding_idx=0)  
        self.fc1 = torch.nn.Linear(embed_dim*2, 1)
        torch.nn.init.xavier_uniform_(self.embed1.weight.data[1:, :])
        #torch.nn.init.uniform_(self.embed1.weight.data[1:, :], 0, 1./np.sqrt(self.inputSize*1.))


    def forward(self, ctx, itm, pos, ctx_v, z):  # ctx: context, itm: item, x3: position, ctx_v: context value
        if not self.training:  # ignore one-hot feature never seen in training phase
            ctx[ctx>=self.inputSize] = 0
            ctx_v[ctx>=self.inputSize] = 0
        ctx = torch.sum(torch.mul(self.embed1(ctx), ctx_v.unsqueeze(2)), dim=1)  # field 1 embedding for cxt: (batch_size, cxt_nonzero_feature_num, embed_dim)
        itm = torch.sum(self.embed1(itm), dim=1)  # field 1 embedding for item: (batch_size, item_nonzero_feature_num, embed_dim)

        ## merge
        x12 = self.fc1(torch.cat((ctx*itm, z), 1))  # (batch_size,)
        #print('G size', x12.size())
        return x12

class Discriminator(torch.nn.Module):
    def __init__(self, inputSize, embed_dim):
        super().__init__()
        self.inputSize = inputSize
        self.embed1 = torch.nn.Embedding(inputSize, embed_dim, padding_idx=0)  
        self.l1 = torch.nn.Linear(embed_dim+1, 1)

        torch.nn.init.xavier_uniform_(self.embed1.weight.data[1:, :])
        torch.nn.init.xavier_uniform_(self.l1.weight.data)

    def forward(self, ctx, itm, fake_y, pos, ctx_v):  # ctx: context, itm: item, x3: position, ctx_v: context value
        if not self.training:  # ignore one-hot feature never seen in training phase
            ctx[ctx>self.inputSize] = 0
            ctx_v[ctx>self.inputSize] = 0
        ctx = torch.sum(torch.mul(self.embed1(ctx), ctx_v.unsqueeze(2)), dim=1)  # field 1 embedding for cxt: (batch_size, cxt_nonzero_feature_num, embed_dim)
        itm = torch.sum(self.embed1(itm), dim=1)  # field 1 embedding for item: (batch_size, item_nonzero_feature_num, embed_dim)

        ## merge
        x12 = ctx*itm  # (batch_size, embed_dim)
        output = self.l1(torch.cat((x12, fake_y.unsqueeze(1)), 1))
        #print('D size', output.size())
        return output.squeeze(1)




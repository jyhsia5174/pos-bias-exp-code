import torch

from src.model.layer import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron


class DeepCrossNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.

    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, input_dims, embed_dim, feats_num, num_layers, mlp_dims, dropout):
        super().__init__()
        #self.embedding = FeaturesEmbedding(input_dims, embed_dim)
        self.embedding = torch.nn.Embedding(input_dims, embed_dim, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data[1:, :])

        self.embed_output_dim = feats_num * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x1, x2, x3):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        #embed_x = self.embedding(x1, x2).view(-1, self.embed_output_dim)
        embed1 = torch.mul(self.embedding(x1), x3.unsqueeze(2))#.mean(dim=1, keepdim=True)
        embed2 = self.embedding(x2)#.mean(dim=1, keepdim=True)
        embed_x = torch.cat((embed1, embed2), 1)
        embed_x = embed_x.view(embed_x.size()[0], -1)
        #print(embed_x.size)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        return torch.sigmoid(p.squeeze(1))

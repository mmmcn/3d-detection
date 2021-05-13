import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        # use dropout or not
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # x: (B, C, num_points, num_samples)
        x = x.permute(0, 2, 3, 1)
        # x = x + self.norm(self.dropout(sublayer(x)))
        # residual connection ---> layer norm ---> dropout
        x = self.dropout(self.norm(x + sublayer(x)))

        return x.permute(0, 3, 1, 2).contiguous()
        # return x + self.dropout(sublayer(self.norm(x)))

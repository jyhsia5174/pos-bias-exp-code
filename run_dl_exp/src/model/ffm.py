import torch

from src.model.layer import FeaturesLinear, FieldAwareFactorizationMachine


class FieldAwareFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Field-aware Factorization Machine.

    Reference:
        Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x_field, x, x_val=None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x_field, x, x_val), dim=1), dim=1, keepdim=True)
        #x = self.linear(x_field, x, x_val) + ffm_term
        x = ffm_term
        return x.squeeze(1)

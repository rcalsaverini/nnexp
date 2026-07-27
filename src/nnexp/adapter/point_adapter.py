import torch
import torch.nn as nn


class PointAdapter(nn.Module):
    """
    Neural projection adapter converting continuous 2D point cloud matrices X in R^(B x 2 x N)
    into transformer embedding vectors E_points in R^(B x N x d_model).
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_dim: int = 256,
        d_model: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.d_model = d_model

        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting continuous coordinate pairs (x_i, y_i) to embeddings.

        Args:
            point_cloud: Tensor of shape [B, 2, N] or [B, N, 2]

        Returns:
            E_points: Continuous point embedding tensor of shape [B, N, d_model]
        """
        if point_cloud.dim() == 3 and point_cloud.shape[1] == 2 and point_cloud.shape[2] != 2:
            point_cloud = point_cloud.transpose(1, 2)  # [B, N, 2]

        h = self.input_layer(point_cloud)
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)
        e_points = self.output_layer(h)
        e_points = self.norm2(e_points)
        return e_points

    @staticmethod
    def concat_embeddings(E_prompt: torch.Tensor, E_points: torch.Tensor) -> torch.Tensor:
        """
        Concatenates prompt embeddings and continuous point cloud embeddings.

        Args:
            E_prompt: Text prompt embedding tensor of shape [B, L_prompt, d_model]
            E_points: Point cloud embedding tensor of shape [B, N, d_model]

        Returns:
            E_total: Total input embedding tensor of shape [B, L_prompt + N, d_model]
        """
        return torch.cat([E_prompt, E_points], dim=1)

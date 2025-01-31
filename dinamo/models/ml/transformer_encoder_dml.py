import torch.nn as nn


class TransformerEncoderDML(nn.Module):
    def __init__(
            self, 
            n_bins=100, 
            d_model=32, 
            nhead=4, 
            num_layers=2, 
            dim_feedforward=64, 
            dropout=0.1
        ):
        super().__init__()
        self.n_bins = n_bins
        
        self.x_emb = nn.Sequential(
            nn.Linear(n_bins, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.t_emb = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

        self.μ_out = nn.Sequential(
            nn.Linear(d_model, n_bins),
            nn.LayerNorm(n_bins),
            nn.ReLU(),
            nn.Linear(n_bins, n_bins),
            nn.Softmax(dim=1) # normalized to unity
        )
        self.σ_out = nn.Sequential(
            nn.Linear(d_model, n_bins),
            nn.LayerNorm(n_bins),
            nn.ReLU(),
            nn.Linear(n_bins, n_bins),
            nn.Softplus() # only positive values
        )

    def forward(self, x, t):
        """
        x: (K, M, n_bins)
        t: (K, M,)
        => outputs μ, σ; each shape of (K, n_bins)
        """
        x_embed = self.x_emb(x) # (K, M, n_bins) -> (K, M, d_model)
        t = t.unsqueeze(2) # (K, M,) -> (K, M, 1)
        t_embed = self.t_emb(t) # (K, M, 1) -> (K, M, d_model)

        xt = x_embed + t_embed # (K, M, d_model)
        xt_e = self.transformer_encoder(xt) # (K, M, d_model)

        # average pool
        xt_e_pooled = xt_e.mean(dim=1) # (K, d_model)

        μ = self.μ_out(xt_e_pooled) # (K, d_model) -> (K, n_bins)        
        σ = self.σ_out(xt_e_pooled) # (K, d_model) -> (K, n_bins)
        return μ, σ

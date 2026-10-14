import torch
from torch import nn

from einops import rearrange

class Regressor(nn.Module):
    """
    Regression-based model wrapper
    """

    def __init__(
        self,
        backbone: nn.Module,
        query_model: nn.Module,
        query_adapter: nn.Module,
        loss: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.query_model = query_model
        self.query_adapter = query_adapter
        self.loss_fn = loss

    def forward(self, x, queries, query_is_embedding: bool = False,
                level=None):
        """
        Args:
            x: torch.Tensor
                Mixture signal (B, C, T)

            queries: torch.Tensor
                Audio query segments (B, C, T_em), or -- if query_is_embedding --
                query embeddings ALREADY IN PCA SPACE (B, D_pca).

        Returns:
            s: torch.Tensor
                Separated source stems (B, C, T)
        """

        # Separator and SEG generator share one query space (PaSST ->
        # PcaAdapter -> D_pca), so a precomputed embedding skips both steps.
        if not query_is_embedding:
            queries = self.query_adapter(self.query_model(queries))

        # Only forward `level` to backbones that declare `accepts_level`;
        # asking an unlevelled backbone for a level is a caller error.
        if getattr(self.backbone, "accepts_level", False):
            return self.backbone(x, queries, level=level)
        if level is not None:
            raise TypeError(
                f"{type(self.backbone).__name__} does not accept `level` "
                f"(accepts_level is not set), but one was passed")
        return self.backbone(x, queries)

    def train_step(self, s, q, debug: bool = False, *args, **kwargs):
        """
        Args:
            s: torch.Tensor
                Source stems (B, N_src, C, T)

            q: torch.Tensor
                Audio query segments for embedding (B, C, T_em)

        Returns:
            loss_dict: dict
                Losses in a dictionary {loss, loss_linear, loss_unnorm}
        """

        mix = s.sum(dim=1)

        pred = self.forward(mix, q)
        target = s[:, 0, :]

        loss_unnorm = (pred - target).pow(2).mean(dim=(1, 2))
        loss_norm = loss_unnorm / (target.pow(2).mean(dim=(1, 2)) + 1e-8)

        loss_linear = loss_norm.mean()
        loss_unnorm = loss_unnorm.mean()

        loss = self.loss_fn(pred, target)

        return {
            "loss": loss,
            "loss_linear": loss_linear,
            "loss_unnorm": loss_unnorm,
        }

    @torch.no_grad()
    def sample(
        self,
        queries,
        mix: torch.Tensor | None = None,
        sources: torch.Tensor | None = None,
        debug: bool = False,
        *args,
        **kwargs,
    ):
        """
        queries: (B, N_src, C, T), or (B, N_src, D_orig_emb) if query_is_embedding
        mix: (B, C, T)
        sources: (B, N_src, C, T)
        """
        query_is_embedding = kwargs.pop("query_is_embedding", False)
        level = kwargs.pop("level", None)
        assert mix is None or sources is None, "Provide exactly one of mix or sources."

        if sources is not None:
            mix = sources.sum(dim=1)  # (B, C, T)

        assert mix is not None, "Provide either mix or sources."

        # Expand batch to all sources and rearrange to merge batch and source dims
        b, n = queries.shape[:2]  # b=batch_size, n=n_src
        mix = mix.unsqueeze(1)  # [B, 1, C, T]
        mix = mix.expand((-1, n, -1, -1))  # [B, N, C, T]
        mix = rearrange(mix, "b n c t -> (b n) c t")  # [B*N, C, T]
        queries = queries.reshape((b * n,) + queries.shape[2:])  # [B*N, ...]

        # `level` arrives per ITEM, [B]; the forward runs on the flattened
        # [B*N] batch, so it has to be repeated the same way the mixture was
        # expanded -- interleaved, because the flattening is (b n), not (n b).
        if level is not None:
            level = level.reshape(-1)
            if level.shape[0] == b:
                level = level.repeat_interleave(n)
            elif level.shape[0] != b * n:
                raise ValueError(f"level has {level.shape[0]} entries, "
                                 f"expected {b} or {b * n}")

        x = self.forward(mix, queries, query_is_embedding=query_is_embedding,
                         level=level)

        # unchunk batch and source dimension
        x = rearrange(x, "(b n) c t -> b n c t", n=n, b=b)


        if debug and sources is not None:
            info = {}
            info["sources"] = sources
            return x, info

        return x

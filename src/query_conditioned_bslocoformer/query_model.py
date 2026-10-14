import pickle

import torch
import torch.nn.functional as F
import torchaudio as ta
from einops import rearrange
from hear21passt.base import get_basic_model
from torch import nn


class TorchPCA(nn.Module):
    """PyTorch implementation of PCA projection using sklearn's parameters."""

    def __init__(self, mean, components):
        """
        Args:
            mean: np.array of shape (n_features,) - the data mean
            components: np.array of shape (n_components, n_features) - principal components
        """
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("components", torch.from_numpy(components).float())

    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (..., n_features)
        Returns:
            torch.Tensor of shape (..., n_components)
        """
        # Center and project: (x - mean) @ components.T
        return (x - self.mean) @ self.components.T


class PcaAdapter(nn.Module):
    """Adapter class for TorchPCA that loads parameters for the pca."""

    def __init__(self, pca_pkl="src/query_conditioned_bslocoformer/pca_params_train.pkl"):
        """
        Args:
            pca_pkl: str | Path - path to pickle file containing mean and components
        """
        super().__init__()
        # Kept so a consumer can ask which basis this model emits in (see
        # `StemClassifier.check_basis`).
        self.pca_pkl = str(pca_pkl)
        with open(pca_pkl, "rb") as f:
            pca_params = pickle.load(f)
            self.pca = TorchPCA(pca_params["mean"], pca_params["components"])

    def forward(self, x):
        return self.pca(x)


def _patch_passt_attention():
    """Make hear21passt use flash attention. Idempotent, applied once.

    `hear21passt.models.passt.Attention` predates SDPA and materialises the
    full attention score matrix with a manual softmax; swapping in
    `scaled_dot_product_attention` is a large speedup with negligible
    (bf16-rounding-level) numerical difference.

    Patches the library class rather than subclassing it, since the
    checkpoint loader builds these modules itself inside `get_basic_model`.
    """
    from hear21passt.models import passt as _passt

    if getattr(_passt.Attention, "_sdpa_patched", False):
        return

    def forward(self, x):
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        out = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], dropout_p=self.attn_drop.p if self.training else 0.0
        )
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, n, c)))

    _passt.Attention.forward = forward
    _passt.Attention._sdpa_patched = True


class PasstWrapper(nn.Module):
    PASST_EMB_DIM: int = 768
    PASST_FS: int = 32000

    def __init__(
        self,
        cond_emb_dim: int = 384,
        original_fs: int = 44100,
        passt_fs: int = PASST_FS,
    ):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim

        _patch_passt_attention()
        self.passt = get_basic_model(mode="embed_only", arch="openmic").eval()
        self.resample = ta.transforms.Resample(
            orig_freq=original_fs, new_freq=passt_fs
        ).eval()

        for p in self.passt.parameters():
            p.requires_grad = False

    @torch.no_grad()
    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, qaudio):
        """
        Forward pass of the PasstWrapper model.

        Args:
            qaudio: torch.Tensor
                Query audio (B, C, T) where N is the number of query sources

        Returns:
            embeddings: torch.Tensor
                Embeddings (B, D_emb)
        """
        x = torch.mean(qaudio, dim=1)
        x = self.resample(x)

        specs = self.passt.mel(x)[..., :998]
        specs = specs[:, None, ...]
        _, z = self.passt.net(specs)

        return z

    @torch.no_grad()
    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def embed(self, qaudio):
        """
        Embed audio without PCA; use for fitting the PCA.

        Args:
            qaudio: torch.Tensor
                Query audio (B, N, C, T) where N is the number of query sources

        Returns:
            embeddings: torch.Tensor
                Embeddings (B*N, D_orig_emb)
        """
        x = rearrange(qaudio, "b n c t -> (b n) c t")
        x = torch.mean(x, dim=1)
        x = self.resample(x)

        specs = self.passt.mel(x)[..., :998]
        specs = specs[:, None, ...]
        _, z = self.passt.net(specs)

        return z

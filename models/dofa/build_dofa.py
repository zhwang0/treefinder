import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .dofa_models_dwv import vit_base_patch16, vit_base_patch16_segmentation


class DOFAModel(nn.Module):
    """
    DOFA ViT-based segmentation wrapper.
    """

    def __init__(self, num_classes, image_size, wave_list, pretrained_path):
        super(DOFAModel, self).__init__()
        self.wave = wave_list
        self.num_features = len(wave_list)

        # Load backbone and pretrained weights
        self.vit = vit_base_patch16()
        if pretrained_path:
            info = self.vit.load_state_dict(
                torch.load(pretrained_path), strict=False
            )
            print(
                f"[Backbone] loaded — missing: {info.missing_keys}, "
                f"unexpected: {info.unexpected_keys}"
            )

        # Build segmentation model
        self.seg = vit_base_patch16_segmentation(
            num_classes=num_classes,
            img_size=image_size
        )

        # Transfer compatible weights from backbone to segmentation model
        self._transfer_weights()

    def _transfer_weights(self):
        """Filter ViT weights and interpolate positional embeddings for segmentation."""
        vit_dict = self.vit.state_dict()
        seg_dict = self.seg.state_dict()

        # Keep matching keys, exclude heads
        filtered = {
            k: v for k, v in vit_dict.items()
            if k in seg_dict and not k.startswith("head.") and not k.startswith("fc_norm.")
        }

        # # Interpolate pos_embed if needed
        # if "pos_embed" in filtered:
        #     filtered["pos_embed"] = self._interpolate_pos_embed(
        #         filtered["pos_embed"],
        #         new_num_patches=self.seg.num_patches,
        #         embed_dim=self.seg.pos_embed.shape[-1]
        #     )

        # Update and load into segmentation model
        seg_dict.update(filtered)
        info = self.seg.load_state_dict(seg_dict)
        print(
            f"[Segmentation] loaded — missing: {info.missing_keys}, "
            f"unexpected: {info.unexpected_keys}"
        )

    @staticmethod
    def _interpolate_pos_embed(pos_embed: torch.Tensor, new_num_patches: int, embed_dim: int) -> torch.Tensor:
        """Resize transformer positional embeddings to a new grid size."""
        # Exclude CLS token
        orig_tokens = pos_embed[:, 1:, :]
        orig_n = orig_tokens.shape[1]
        orig_grid = int(orig_n ** 0.5)
        new_grid = int(new_num_patches ** 0.5)

        # Reshape to (1, C, H, W)
        tokens = orig_tokens.reshape(1, orig_grid, orig_grid, embed_dim).permute(0, 3, 1, 2)

        # Bilinear interpolate
        tokens = F.interpolate(
            tokens,
            size=(new_grid, new_grid),
            mode="bilinear",
            align_corners=False
        )

        # Reshape back and prepend CLS
        tokens = tokens.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, embed_dim)
        return torch.cat([pos_embed[:, :1, :], tokens], dim=1)


    def forward(self, x):
        """
        Forward pass: returns segmentation logits (B, C, H, W).

        Args:
            x: input tensor (B, C, H, W)
        """
        return self.seg(x[:, :self.num_features], self.wave)


def build_dofa(cfg: dict) -> nn.Module:
    img_size    = int(cfg.get('image_size', 224))
    num_classes = int(cfg.get('num_classes', 1))
    pretrained_path = cfg['dofa_pretrained'].get('dofa_weights', 'dofa/DOFA_ViT_base_e100.pth')
    pretrained_path = Path(cfg.get('pretrain_dir', 'pretrained_weights')) / pretrained_path
    wave_list  = cfg['dofa_pretrained'].get('dofa_wavelist', [0.65, 0.55, 0.45, 0.85])
    wave_list = [float(w) for w in wave_list]
    return DOFAModel(
        num_classes=num_classes,
        image_size=img_size,
        wave_list=wave_list,
        pretrained_path=pretrained_path
    )
    
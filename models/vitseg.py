import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

# # Custom ViTPatchEmbeddings with 6 input channels
# class CustomViTPatchEmbeddings(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.num_channels = config.num_channels
#         self.patch_size = config.patch_size
#         self.projection = nn.Conv2d(
#             in_channels=self.num_channels,
#             out_channels=config.hidden_size,
#             kernel_size=config.patch_size,
#             stride=config.patch_size
#         )
#         self.num_patches = (config.image_size // config.patch_size) ** 2

#     def forward(self, pixel_values, interpolate_pos_encoding=False):
#         batch_size, num_channels, height, width = pixel_values.shape
#         if num_channels != self.num_channels:
#             raise ValueError(
#                 f"Expected {self.num_channels} channels but got {num_channels}."
#             )
#         embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
#         return embeddings

class ViTSegmentation(nn.Module):
    def __init__(self, num_classes, image_size, num_channels, patch_size, pretrain_name):
        super(ViTSegmentation, self).__init__()
        
        self.num_patches = (image_size // patch_size) ** 2
        
        # init empty model
        self.config = ViTConfig.from_pretrained(pretrain_name)
        self.config.image_size = image_size
        self.config.num_labels = num_classes
        self.config.num_channels = num_channels
        self.vit = ViTModel(self.config)
        
        
        # Load pre-trained weights
        pretrained_model = ViTModel.from_pretrained(
            pretrain_name,
            ignore_mismatched_sizes=True)
        
        # Copy weights to the pre-trained 3-channel weights to initialize the x-channel input weights
        with torch.no_grad():
            original_proj_weight = pretrained_model.embeddings.patch_embeddings.projection.weight
            pretrained_model.embeddings.patch_embeddings.projection.weight = nn.Parameter(
                torch.cat([original_proj_weight] * 3, dim=1)[:, :num_channels, :, :]
            )
        
        
        # update the model's state_dict with the modified pretrained weights
        model_dict = self.vit.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.vit.load_state_dict(model_dict, strict=False)
        

        # # Interpolate positional embeddings for the new image size if it differs from 224x224
        # self.num_patches = (image_size // patch_size) ** 2
        # self.vit.embeddings.position_embeddings = nn.Parameter(
        #     self.interpolate_pos_embed(pretrained_model.embeddings.position_embeddings, self.num_patches, self.config.hidden_size)
        # )

        # Decoder to upsample to original resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.config.hidden_size, 256, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Upsample to 512x512
            nn.Conv2d(64, num_classes, kernel_size=1)  # Final output layer with num_classes channels
        )
    
    def interpolate_pos_embed(self, pos_embed, num_patches, embed_dim):
        # Interpolate positional embeddings to match new patch count
        num_patches_original = pos_embed.shape[1] - 1  # Exclude CLS token
        grid_size_original = int(num_patches_original ** 0.5)
        grid_size_new = int(num_patches ** 0.5)
        
        pos_tokens = pos_embed[:, 1:, :].reshape(1, grid_size_original, grid_size_original, embed_dim).permute(0, 3, 1, 2)
        pos_tokens_resized = torch.nn.functional.interpolate(
            pos_tokens, size=(grid_size_new, grid_size_new), mode="bilinear", align_corners=False
        )
        pos_tokens_resized = pos_tokens_resized.permute(0, 2, 3, 1).reshape(1, grid_size_new * grid_size_new, embed_dim)
        return torch.cat([pos_embed[:, :1, :], pos_tokens_resized], dim=1)


    def forward(self, x):
        # Pass input through ViT to obtain 32x32 feature map
        features = self.vit(x).last_hidden_state  # Shape: (batch_size, num_patches+1, hidden_size)
        features = features[:, 1:, :]  # Remove CLS token
        h, w = int(self.num_patches ** 0.5), int(self.num_patches ** 0.5)
        features = features.transpose(1, 2).reshape(-1, self.config.hidden_size, h, w)  # Shape: (batch_size, hidden_size, 32, 32)

        # Upsample to 512x512 using the decoder
        segmentation_output = self.decoder(features)  # Shape: (batch_size, num_classes, 512, 512)
        
        return segmentation_output


def build_vit(cfg: dict) -> nn.Module:
    img_size    = int(cfg.get('image_size', 224))
    in_ch       = int(cfg.get('in_channels', 5))
    num_classes = int(cfg.get('num_classes', 1))
    pretrain_name = cfg['vit_pretrained'].get('vit_weights', 'google/vit-base-patch16-224-in21k')
    patch_size  = int(cfg['vit_pretrained'].get('vit_patch_size', 16))
    return ViTSegmentation(
        num_classes=num_classes,
        image_size=img_size,
        num_channels=in_ch,
        patch_size=patch_size,
        pretrain_name=pretrain_name
    )




class ViTSegmentation3(nn.Module):
    def __init__(self, num_classes, image_size, num_channels, patch_size, pretrain_name):
        super(ViTSegmentation3, self).__init__()
        
        self.num_patches = (image_size // patch_size) ** 2
        
        # init empty model
        self.config = ViTConfig.from_pretrained(pretrain_name)
        self.config.image_size = image_size
        self.config.num_labels = num_classes
        self.config.num_channels = num_channels
        self.vit = ViTModel(self.config)
        
        
        # Load pre-trained weights
        pretrained_model = ViTModel.from_pretrained(
            pretrain_name,
            ignore_mismatched_sizes=True)
        
        # # Copy weights to the pre-trained 3-channel weights to initialize the x-channel input weights
        # with torch.no_grad():
        #     original_proj_weight = pretrained_model.embeddings.patch_embeddings.projection.weight
        #     pretrained_model.embeddings.patch_embeddings.projection.weight = nn.Parameter(
        #         torch.cat([original_proj_weight] * 3, dim=1)[:, :num_channels, :, :]
        #     )
        
        
        # update the model's state_dict with the modified pretrained weights
        model_dict = self.vit.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.vit.load_state_dict(model_dict, strict=False)
        

        # # Interpolate positional embeddings for the new image size if it differs from 224x224
        # self.num_patches = (image_size // patch_size) ** 2
        # self.vit.embeddings.position_embeddings = nn.Parameter(
        #     self.interpolate_pos_embed(pretrained_model.embeddings.position_embeddings, self.num_patches, self.config.hidden_size)
        # )

        # Decoder to upsample to original resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.config.hidden_size, 256, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Upsample to 512x512
            nn.Conv2d(64, num_classes, kernel_size=1)  # Final output layer with num_classes channels
        )
    
    def interpolate_pos_embed(self, pos_embed, num_patches, embed_dim):
        # Interpolate positional embeddings to match new patch count
        num_patches_original = pos_embed.shape[1] - 1  # Exclude CLS token
        grid_size_original = int(num_patches_original ** 0.5)
        grid_size_new = int(num_patches ** 0.5)
        
        pos_tokens = pos_embed[:, 1:, :].reshape(1, grid_size_original, grid_size_original, embed_dim).permute(0, 3, 1, 2)
        pos_tokens_resized = torch.nn.functional.interpolate(
            pos_tokens, size=(grid_size_new, grid_size_new), mode="bilinear", align_corners=False
        )
        pos_tokens_resized = pos_tokens_resized.permute(0, 2, 3, 1).reshape(1, grid_size_new * grid_size_new, embed_dim)
        return torch.cat([pos_embed[:, :1, :], pos_tokens_resized], dim=1)


    def forward(self, x):
        # Pass input through ViT to obtain 32x32 feature map
        features = self.vit(x[:,:3]).last_hidden_state  # Shape: (batch_size, num_patches+1, hidden_size)
        features = features[:, 1:, :]  # Remove CLS token
        h, w = int(self.num_patches ** 0.5), int(self.num_patches ** 0.5)
        features = features.transpose(1, 2).reshape(-1, self.config.hidden_size, h, w)  # Shape: (batch_size, hidden_size, 32, 32)

        # Upsample to 512x512 using the decoder
        segmentation_output = self.decoder(features)  # Shape: (batch_size, num_classes, 512, 512)
        
        return segmentation_output


def build_vit3(cfg: dict) -> nn.Module:
    img_size    = int(cfg.get('image_size', 224))
    in_ch       = int(cfg.get('in_channels', 5))
    num_classes = int(cfg.get('num_classes', 1))
    pretrain_name = cfg['vit_pretrained'].get('vit_weights', 'google/vit-base-patch16-224-in21k')
    patch_size  = int(cfg['vit_pretrained'].get('vit_patch_size', 16))
    return ViTSegmentation3(
        num_classes=num_classes,
        image_size=img_size,
        num_channels=3,
        patch_size=patch_size,
        pretrain_name=pretrain_name
    )
    
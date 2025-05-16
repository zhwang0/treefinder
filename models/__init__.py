import torch.nn as nn

def get_model(model_cfg: dict) -> nn.Module:
    name = model_cfg.get('name', 'unet')
    
    if name.lower() == 'unet':
        from .unet import build_unet
        return build_unet(model_cfg)
    
    elif name in ('deeplabv3','deeplabv3+'):
        from .deeplab import build_deeplabv3
        return build_deeplabv3(model_cfg)
    
    elif name in ('d3_tf'):
        from .deeplab import build_deeplab_plus_tf
        return build_deeplab_plus_tf(model_cfg)
    
    elif name in ('vitseg', 'vit_segmentation'):
        from .vitseg import build_vit
        return build_vit(model_cfg)
    
    elif name in ('vitseg3'):
        from .vitseg import build_vit3
        return build_vit3(model_cfg)
    
    elif name in ('segformer', 'segformer_segmentation'):
        from .segformer import build_segformer
        return build_segformer(model_cfg)
    
    elif name in ('segformer3'):
        from .segformer import build_segformer3
        return build_segformer3(model_cfg)
    
    elif name in ('segformer_large3'):
        from .segformer import build_segformer_large3
        return build_segformer_large3(model_cfg)
    
    elif name in ('mask2former'):
        from .mask2former import build_mask2former
        return build_mask2former(model_cfg)
    
    elif name in ('dofa', 'dofa_segmentation'):
        from .dofa.build_dofa import build_dofa
        return build_dofa(model_cfg)
    
    raise ValueError(f"Unsupported model: {name}")
import os
import time
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.ops import sigmoid_focal_loss

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)          # shape (B,1,H,W)
        if mask is not None:
            probs = probs * mask
            
        p_flat = probs.view(-1)
        t_flat = targets.float().view(-1)
        
        inter = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        dice = (inter + self.eps) / (union + self.eps)
        return 1 - dice
    
class FocalLoss(nn.Module):
  def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction  # 'none' is recommended for external masking

  def forward(self, logits, targets):
    return sigmoid_focal_loss(
      inputs=logits,
      targets=targets.float(),
      alpha=self.alpha,
      gamma=self.gamma,
      reduction=self.reduction
    )
    

def train_model(model, train_loader, val_loader, cfg, exp_name):
    """
    Train the model with validation and early stopping.

    Args:
      model: nn.Module
      train_loader: DataLoader for training
      val_loader: DataLoader for validation
      cfg: full config dict
      exp_name: experiment identifier

    Returns:
      dict containing best monitored metric
    """
    logger = logging.getLogger(__name__)
    # Device
    gpu_id = cfg['experiment'].get('gpu_id', 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # Unpack training cfg
    tr_cfg = cfg['training']
    criterion_type = tr_cfg['criterion'].get('type', 'BCEWithLogitsLoss')
    if criterion_type == 'BCEWithLogitsLoss':
        if tr_cfg['criterion'].get('w_pos', 0) > 0:
            pos_weight = torch.tensor(tr_cfg['criterion']['w_pos'], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        else:
            criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif criterion_type == 'FocalLoss':
        alpha = tr_cfg['criterion'].get('alpha', 0.25)
        gamma = tr_cfg['criterion'].get('gamma', 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unsupported loss type: {criterion_type}")
    dice_loss = DiceLoss()
    w_dice = tr_cfg.get('w_dice', 0.5)
    
    optimizer = getattr(optim, tr_cfg['optimizer']['type'])(
        model.parameters(),
        lr=float(tr_cfg['learning_rate']),
        weight_decay=tr_cfg['optimizer'].get('weight_decay', 0)
    )
    scheduler = None
    if 'scheduler' in tr_cfg:
        sc = tr_cfg['scheduler']        
        sched_type = sc['type']
        SchedulerClass = getattr(optim.lr_scheduler, sched_type)
        
        if sched_type == 'ExponentialLR':
            scheduler = SchedulerClass(
            optimizer,
            gamma=sc.get('gamma', 0.95)
        )
        elif sched_type == 'StepLR':
            scheduler = SchedulerClass(
            optimizer,
            step_size=sc.get('step_size', 10),
            gamma=sc.get('gamma', 0.1)
        )
        elif sched_type == 'CosineAnnealingWarmRestarts':
            scheduler = SchedulerClass(
            optimizer,
            T_0=sc.get('T_0', 10),
            T_mult=sc.get('T_mult', 1),
            eta_min=float(sc.get('eta_min', 0))
        )
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")


    # Early stopping config
    es_cfg = tr_cfg['early_stopping']
    es_enabled = es_cfg.get('enabled', False)
    monitor = es_cfg.get('monitor', 'val_loss')
    mode = es_cfg.get('mode', 'min')
    patience = es_cfg.get('patience', 5)
    if mode == 'min':
        best_val = float('inf')
        improve = lambda cur, best: cur < best
    else:
        best_val = -float('inf')
        improve = lambda cur, best: cur > best
    epochs_no_improve = 0

    # Checkpoint directory
    ckpt_root = Path(cfg['output']['checkpoint_dir'])
    ckpt_dir = ckpt_root / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results_root = Path(cfg['output']['results_dir'])
    results_dir = results_root / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = tr_cfg['epochs']
    log_interval = cfg['logging'].get('log_interval', 100)
    
    best_metric = None
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss_total = 0.0
        train_loss_bce   = 0.0
        train_loss_dice  = 0.0
        # train_loss_focal = 0.0
        step_times   = []
        for step, batch in enumerate(train_loader, 1):
            step_start = time.time()
            imgs = batch['image'].to(device)
            labels = batch['label'].unsqueeze(1).float().to(device)
            no_data = batch['no_data_mask'].unsqueeze(1).to(device)
            valid_mask = (~no_data).float()
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            raw_loss = criterion(outputs, labels) # (B,1,H,W)
            bce_loss = (raw_loss * valid_mask).sum() / valid_mask.sum()
            if w_dice > 0:
                dloss = dice_loss(outputs, labels, valid_mask)
            else: 
                dloss = torch.tensor(0.0, device=device)
                
            total_loss = bce_loss + w_dice * dloss
            total_loss.backward()
            optimizer.step()
            
            
            train_loss_total += total_loss.item()
            train_loss_bce += bce_loss.item()
            train_loss_dice += dloss.item()
            step_times.append(time.time() - step_start)
            
            if step % log_interval == 0:
                avg_total = train_loss_total / step
                avg_bce = train_loss_bce / step
                avg_dice = train_loss_dice / step
                log_str = (
                    f"[Epoch {epoch}/{num_epochs}] "
                    f"Step {step}/{len(train_loader)} - "
                    f"Time Spent: {sum(step_times)/60:.1f}m - "
                    f"Train Total Loss: {avg_total:.4f}"
                )
                if w_dice > 0:
                    log_str += f" - BCE Loss: {avg_bce:.4f} - Dice Loss: {avg_dice:.4f}"
                logger.info(log_str)
                step_times = []

        avg_train = train_loss_total / len(train_loader)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_loss_bce = 0.0
        val_loss_dice = 0.0
        val_time = time.time()
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                labels = batch['label'].unsqueeze(1).float().to(device)
                no_data = batch['no_data_mask'].unsqueeze(1).to(device)
                valid_mask = (~no_data).float()
                
                outputs = model(imgs)
                
                raw_loss = criterion(outputs, labels)
                bce_loss = (raw_loss * valid_mask).sum() / valid_mask.sum()
                if w_dice > 0:
                    dloss = dice_loss(outputs, labels, valid_mask)
                else:
                    dloss = torch.tensor(0.0, device=device)
                total_loss = bce_loss + w_dice * dloss
                
                val_loss_total += total_loss.item()
                val_loss_bce += bce_loss.item()
                val_loss_dice += dloss.item()
                
        avg_val = val_loss_total / len(val_loader)
        avg_bce = val_loss_bce / len(val_loader)
        avg_dice = val_loss_dice / len(val_loader)
        
        val_losses.append(avg_val)
        val_time = time.time() - val_time
        log_str = (
            f"[Epoch {epoch}/{num_epochs}] "
            f"Time Spent: {val_time/60:.1f}m - "
            f"Avg Val Total Loss: {avg_val:.4f}"
            )
        if w_dice > 0:
            log_str += f" - BCE Loss: {avg_bce:.4f} - Dice Loss: {avg_dice:.4f}"
        logger.info(log_str)

        # Save best weights and early stopping check
        current = avg_val if monitor == 'val_loss' else None
        if current is not None and improve(current, best_val):
            best_val = current
            best_path = ckpt_dir / f"{exp_name}_best.pth"
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model ({monitor}={best_val:.4f}) saved: {best_path}")
            best_metric = best_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if es_enabled:
                logger.info(f"No improvement in {monitor} for {epochs_no_improve}/{patience} epochs.")
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Disable epoch checkpoint for saving memory
        # torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pth") 
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} done in {epoch_time/60:.1f}m - Train: {avg_train:.4f}, Val: {avg_val:.4f}")

        
    # plot loss curve
    try:
        epochs = list(range(1, epoch + 1))
        plt.figure()
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        loss_curve_path = results_dir / f"{exp_name}_loss_curve.png"
        plt.savefig(loss_curve_path)
        plt.close()
        logger.info(f"Loss curve saved to {loss_curve_path}")
    except Exception as e:
        logger.warning(f"Failed to plot loss curve: {e}")

    return {f"best_{monitor}": best_metric}


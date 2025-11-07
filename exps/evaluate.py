# File: exps/evaluate.py
import logging
from pathlib import Path
import torch
import yaml
import numpy as np
from tqdm import tqdm


def compute_iou(pred, target):
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-6)


def compute_precision(pred, target):
    tp = np.logical_and(pred == 1, target == 1).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    return tp / (tp + fp + 1e-6)


def compute_recall(pred, target):
    tp = np.logical_and(pred == 1, target == 1).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()
    return tp / (tp + fn + 1e-6)


def compute_f1(pred, target):
    p = compute_precision(pred, target)
    r = compute_recall(pred, target)
    return 2 * p * r / (p + r + 1e-6)


def evaluate_model(model, test_loader, cfg, exp_name):
    """
    Run evaluation on the test set, computing segmentation metrics and saving results.

    Args:
      model: trained segmentation model
      test_loader: DataLoader for test set
      cfg: full configuration dict

    Returns:
      dict of averaged metrics
    """
    logger = logging.getLogger(__name__)
    
    # Set device based on config
    gpu_id = cfg['experiment'].get('gpu_id', 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # load best model weights
    ckpt_root = Path(cfg['output']['checkpoint_dir'])
    ckpt_path = ckpt_root / exp_name / f"{exp_name}_best.pth"
    if ckpt_path.exists():
        logger.info(f"Loading model weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(f"Checkpoint not found at {ckpt_path}. Using current model weights.")
    model.to(device)
    model.eval()

    # Evaluation config
    ev = cfg['evaluation']
    metric_names = ev.get('metrics', [])

    # per class state
    stats = {
        0: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        1: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'correct': 0, 'total': 0
    }

    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            imgs    = batch['image'].to(device)
            gt      = batch['label'].unsqueeze(1).cpu().numpy().astype(np.uint8)
            no_data = batch['no_data_mask'].unsqueeze(1).cpu().numpy()
            cls_gt  = batch['cls_label'].cpu().numpy()

            # Model prediction
            outputs = model(imgs)
            probs   = torch.sigmoid(outputs).cpu().numpy()    # [B,1,H,W]
            preds   = (probs > 0.5).astype(np.uint8)

            B = preds.shape[0]
            for i in range(B):
                pred_i = preds[i, 0]
                gt_i   = gt[i, 0]
                valid  = ~no_data[i, 0]
                p_flat = pred_i[valid]
                g_flat = gt_i[valid]
                
                for c in [0, 1]:
                    stats[c]['tp'] += np.logical_and(p_flat == c, g_flat == c).sum()
                    stats[c]['fp'] += np.logical_and(p_flat == c, g_flat != c).sum()
                    stats[c]['fn'] += np.logical_and(p_flat != c, g_flat == c).sum()
                    stats[c]['tn'] += np.logical_and(p_flat != c, g_flat != c).sum()

                stats['correct'] += (p_flat == g_flat).sum()
                stats['total'] += p_flat.size

    # Aggregate results
    results = {}
    def safe_div(a, b): return float(a) / (b + 1e-6)
    
    for c in [0, 1]:
        if 'precision' in metric_names:
            results[f'class{c}_precision'] = safe_div(stats[c]['tp'], stats[c]['tp'] + stats[c]['fp'])
        if 'recall' in metric_names:
            results[f'class{c}_recall'] = safe_div(stats[c]['tp'], stats[c]['tp'] + stats[c]['fn'])
        if 'f1' in metric_names:
            p = safe_div(stats[c]['tp'], stats[c]['tp'] + stats[c]['fp'])
            r = safe_div(stats[c]['tp'], stats[c]['tp'] + stats[c]['fn'])
            results[f'class{c}_f1'] = safe_div(2 * p * r, p + r)
        if 'iou' in metric_names:
            inter = stats[c]['tp']
            union = stats[c]['tp'] + stats[c]['fp'] + stats[c]['fn']
            results[f'class{c}_iou'] = safe_div(inter, union)
    
    # Macro average two classes
    if 'iou' in metric_names:
        results['res_iou_macro'] = 0.5 * (results['class0_iou'] + results['class1_iou'])
    if 'precision' in metric_names:
        results['res_precision_macro'] = 0.5 * (results['class0_precision'] + results['class1_precision'])
    if 'recall' in metric_names:
        results['res_recall_macro'] = 0.5 * (results['class0_recall'] + results['class1_recall'])
    if 'f1' in metric_names:
        results['res_f1_macro'] = 0.5 * (results['class0_f1'] + results['class1_f1'])
    if 'accuracy' in metric_names:
        results['overall_accuracy'] = safe_div(stats['correct'], stats['total'])

    # Log and return
    logger.info(f"Evaluation results:\n{results}")
    return results
        


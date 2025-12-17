import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import cv2
import os

def calculate_accuracy(predict, target, threshold=2.6):
    # predict: mean score
    # target: mean score
    bin_tar = target > threshold
    bin_pre = predict > threshold
    correct = (bin_tar == bin_pre).sum().item()
    acc = correct / target.size(0)
    return correct, acc

def calculate_lcc(target, predict):
    if len(target.shape) > 1:
        target = target.view(-1)
    if len(predict.shape) > 1:
        predict = predict.view(-1)
    predict = predict.cpu().numpy()
    target = target.cpu().numpy()
    if len(predict) < 2: return 0.0
    lcc = np.corrcoef(predict, target)[0, 1]
    return lcc

def calculate_spearmanr(target, predict):
    if len(target.shape) > 1:
        target = target.view(-1)
    if len(predict.shape) > 1:
        predict = predict.view(-1)
    target_list = target.cpu().numpy()
    predict_list = predict.cpu().numpy()
    if len(predict_list) < 2: return 0.0
    rho, _ = stats.spearmanr(predict_list, target_list)
    return rho

def dist2ave(pred_dist):
    # pred_dist: [B, 5]
    # returns: [B]
    device = pred_dist.device
    r = torch.arange(1, 6, device=device, dtype=torch.float32)
    pred_score = torch.sum(pred_dist * r, dim=1)
    return pred_score

def visualize_grad_cam(model, image_tensor, saliency_tensor, save_path):
    # image_tensor: [1, 3, H, W]
    # saliency_tensor: [1, 1, H, W]
    
    model.eval()
    # Enable grad for CAM
    image_tensor.requires_grad = True
    
    # Forward
    score_dist, _ = model(image_tensor, saliency_tensor)
    pred_score = dist2ave(score_dist)
    
    # Backward for Grad-CAM (maximize predicted score)
    model.zero_grad()
    pred_score.backward()
    
    # Get CAM
    cam = model.get_grad_cam(None) # Arguments handled internally via hooks
    
    if cam is None:
        print("Grad-CAM failed: No gradients captured.")
        return

    # Post-process
    cam = cam.detach().cpu().numpy()[0, 0] # [H_feat, W_feat]
    
    # Resize to image size
    img_h, img_w = image_tensor.shape[2], image_tensor.shape[3]
    cam = cv2.resize(cam, (img_w, img_h))
    
    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Original Image (denormalize)
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    img = image_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Overlay
    cam_img = heatmap + img
    cam_img = cam_img / np.max(cam_img)
    
    # Save
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cam_img)
    plt.title("Grad-CAM")
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()
    print(f"Grad-CAM saved to {save_path}")

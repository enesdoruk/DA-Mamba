import torch
import wandb
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data import *
from datetime import datetime
from torch import topk


def tsne_visualization(source_feats, target_feats, iter):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    
    combined_feats = torch.cat((source_feats, target_feats), dim=0)
    tsne = tsne.fit_transform(combined_feats.cpu().numpy())

    x_min, x_max = np.min(tsne, 0), np.max(tsne, 0)
    tsne = (tsne - x_min) / (x_max - x_min)

    source_domain_list = torch.zeros(source_feats.shape[0]).type(torch.LongTensor)
    target_domain_list = torch.ones(target_feats.shape[0]).type(torch.LongTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).to('cuda:0')

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i in range(len(combined_domain_list)): 
        if combined_domain_list[i] == 0:
            colors = 'blue'
        else:
            colors = 'red'
        ax.scatter(tsne[i,0], tsne[i,1], color=colors, )

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    
    mode = f'epoch_{iter}'
    images = wandb.Image(image, caption=mode)
    wandb.log({"TSNE vis": images})



def featmap_visualization(imgs, feats, layer, iter):
    act_tens = feats[layer][0]
    act_gray = torch.sum(act_tens, 0)
    act_gray = act_gray / act_tens.shape[0]
    act_np = act_gray.detach().cpu().numpy()

    fig = plt.figure()
    imgplot = plt.imshow(act_np)

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    
    mode = f'img_{iter}'
    images = wandb.Image(image, caption=mode)
    wandb.log({"Featuremap vis": images})



def predict_visualization(im, gt, boxes):
    mean = np.array(MEANS)

    im = im.cpu().detach().numpy().transpose(1,2,0)

    image = im + mean
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[0], image.shape[1]

    for i in range(boxes.shape[0]):
        xmin = int(boxes[i][0])
        ymin = int(boxes[i][1])
        xmax = int(boxes[i][2])
        ymax = int(boxes[i][3])

        if xmax > w or xmin < 0 or ymax > h or ymin < 0:
            continue

        image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0), 2)
    
    for i in range(gt.shape[0]):
        xmin = int(gt[i][0] * w)
        ymin = int(gt[i][1] * h)
        xmax = int(gt[i][2] * w)
        ymax = int(gt[i][3] * h)

        image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,0,255), 1)
 
    return image


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    cam = np.dot(weight_softmax.reshape(-1), feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size_upsample)
    return cam_img


def actmap_visualization(orig_image, net, feats):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-1].data.detach().cpu().numpy())
    probs = torch.nn.functional.softmax(feats[-1]).data.squeeze()
    class_idx = topk(probs, 1)[1].int()
    cam = returnCAM(feats[-3].detach().cpu().numpy(), weight_softmax, class_idx)
    _, height, width = orig_image.shape
    
    heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + orig_image.detach().cpu().numpy().transpose(1,2,0) * 0.5

    mode = f"_{datetime.now().hour}-{datetime.now().minute}"
    images = wandb.Image(result, caption=mode)
    wandb.log({"Activationmap vis": images})
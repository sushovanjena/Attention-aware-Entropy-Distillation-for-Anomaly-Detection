# Train = (cosine spatial), Test = (cosine)
import argparse
import os
import cv2
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import csv
from distill import *
from evaluate import evaluate
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image

class MVTecDataset(object):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return [Image.open(p).convert('RGB') for p in self.image_list]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx])
        return self.image_list[idx], image

# ----------------------------------------------------------------------------
# Channel-Attention Module
class channel_attention_module(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(ch, ch//ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch//ratio, ch, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x1 = self.mlp(x1)

        x2 = self.max_pool(x).squeeze(-1).squeeze(-1)
        x2 = self.mlp(x2)

        feats = x1 + x2
        feats = self.sigmoid(feats).unsqueeze(-1).unsqueeze(-1)
        refined_feats = x * feats

        return refined_feats

# Spatial-Attention Module
class spatial_attention_module(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)

        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats

        return refined_feats

# CBAM
class cbam(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.ca = channel_attention_module(channel)
        self.sa = spatial_attention_module()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ----------------------------------------------------------------------------

# -----------------------L_intra---------------------------------------------
def structural_distillation(t_feat, s_feat):
    batch_size = s_feat[0].shape[0]
    num_layers = len(s_feat)
    loss_intra = 0.0

    for k in range(num_layers):
        H, W = s_feat[k].shape[2], s_feat[k].shape[3]
        ch = s_feat[k].shape[1]

        # normalise feature maps accross channels
        s_feat[k] = torch.nn.functional.normalize(s_feat[k],dim = (2,3))
        t_feat[k] = torch.nn.functional.normalize(t_feat[k],dim = (2,3))

        # Reshape feature maps to batch_size x channels x (height * width)
        s_reshaped = s_feat[k].reshape(batch_size, ch, H * W)
        t_reshaped = t_feat[k].reshape(batch_size, ch, H * W)

        # Compute affinity matrices
        As = torch.matmul(s_reshaped.transpose(1, 2), s_reshaped)  # Shape: batch_size x (H * W) x (H * W)
        At = torch.matmul(t_reshaped.transpose(1, 2), t_reshaped)  # Shape: batch_size x (H * W) x (H * W)

        loss = torch.sqrt((torch.sum((As - At)**2 , (1,2)))/(H*W*H*W)).mean()

        loss_intra +=  loss

    return loss_intra

# -----------------------L_cd---------------------------------------------
def calculate_cd(t_feat, s_feat):
    distances = []
    for j in range(len(t_feat)):
      cos_sim = 1 - torch.sum(t_feat[j] * s_feat[j], dim=1)/(torch.norm(t_feat[j], p=2, dim=1) * torch.norm(s_feat[j], p=2, dim=1))
      distances.append(cos_sim)

    # Calculate average distance for each feature map
    avg_distances = [distance.mean(dim=(1, 2)) for distance in distances]

    cosine_distance = sum(avg_distance.mean() for avg_distance in avg_distances)

    return cosine_distance

#----------------------L_cd spatial--------------------------------------
def calculate_cd_spatial(t_feat, s_feat):
  loss = 0.
  for i in range(len(t_feat)):
    s_flat = s_feat[i].view(s_feat[i].shape[0],s_feat[i].shape[1],-1)
    t_flat = t_feat[i].view(t_feat[i].shape[0],t_feat[i].shape[1],-1)

    cos_sim = 1 - torch.sum(t_flat * s_flat, dim=2)/((torch.norm(t_flat, p=2, dim=2) * torch.norm(s_flat, p=2, dim=2)) + 1e-6)
    cos_sim = torch.mean(cos_sim, dim=1)

    loss = loss + torch.mean(cos_sim)

  return loss
# -----------------------L_sd---------------------------------------------
def spatial_kl_divergence(F_t, F_s, T):
  num_layers = len(F_s)
  l_sd = 0

  for k in range(num_layers):

    flat_s = F_s[k].view(F_s[k].size(0),F_s[k].size(1), -1)
    flat_t = F_t[k].view(F_t[k].size(0),F_t[k].size(1), -1)

    Phi_F_t = torch.nn.functional.softmax(flat_t/T, dim=-1)
    Phi_F_s = torch.nn.functional.softmax(flat_s/T, dim=-1)

    # Compute KL divergence
    l_sd += torch.sum(torch.sum(Phi_F_t * (torch.log(Phi_F_t) - torch.log(Phi_F_s)), dim=1), dim=1).mean()

  return l_sd

# ------------------------------------------------------------------------------
class ResNet18_MS3(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet18_MS3, self).__init__()
        net = models.resnet18(pretrained=pretrained)
        # ignore the last block and fc
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res



def load_gt(root, cls):
    gt = []
    gt_dir = os.path.join(root, cls, 'ground_truth')
    sub_dirs = sorted(os.listdir(gt_dir))
    for sb in sub_dirs:
        for fname in sorted(os.listdir(os.path.join(gt_dir, sb))):
            temp = cv2.imread(os.path.join(gt_dir, sb, fname), cv2.IMREAD_GRAYSCALE)
            temp = cv2.resize(temp, (256, 256)).astype(np.bool)[None, ...]
            gt.append(temp)
    gt = np.concatenate(gt, 0)
    return gt


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("split", nargs="?", choices=["train", "test"])
    # required training super-parameters
    parser.add_argument("--checkpoint", type=str, default=None, help="student checkpoint")
    parser.add_argument("--category", type=str , default='leather', help="category name for MvTec AD dataset")
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs')

    parser.add_argument("--checkpoint-epoch", type=int, default=100, help="checkpoint resumed for testing (1-based)")
    parser.add_argument("--batch-size", type=int, default=32, help='batch size')
    # trivial parameters
    parser.add_argument("--result-path", type=str, default='results', help="save results")
    parser.add_argument("--save-fig", action='store_true', help="save images with anomaly score")
    parser.add_argument("--mvtec-ad", type=str, default='mvtec_anomaly_detection', help="MvTec-AD dataset path")
    parser.add_argument('--model-save-path', type=str, default='snapshots', help='path where student models are saved')

    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.split == 'train':
        image_list = sorted(glob(os.path.join(args.mvtec_ad, '*/train/good/*.png')))
        train_image_list, val_image_list = train_test_split(image_list, test_size=0.2, random_state=0)
        train_dataset = MVTecDataset(train_image_list, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_dataset = MVTecDataset(val_image_list, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    elif args.split == 'test':
        test_neg_image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'test', 'good', '*.png')))
        test_pos_image_list = set(glob(os.path.join(args.mvtec_ad, args.category, 'test', '*', '*.png'))) - set(test_neg_image_list)
        test_pos_image_list = sorted(list(test_pos_image_list))
        test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
        test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
        test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
        test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)

    teacher = ResNet18_MS3(pretrained=True)
    student = ResNet18_MS3(pretrained=False)
    teacher.cuda()
    student.cuda()

    if args.split == 'train':
        train_val(teacher, student, train_loader, val_loader, args)
    elif args.split == 'test':
        saved_dict = torch.load(args.checkpoint)
        category = args.category
        gt = load_gt(args.mvtec_ad, category)

        print('load ' + args.checkpoint)
        student.load_state_dict(saved_dict['state_dict'])

        pos = test(teacher, student, test_pos_loader)
        neg = test(teacher, student, test_neg_loader)

        scores = []
        for i in range(len(pos)):
            temp = cv2.resize(pos[i], (256, 256))
            scores.append(temp)
        for i in range(len(neg)):
            temp = cv2.resize(neg[i], (256, 256))
            scores.append(temp)

        scores = np.stack(scores)
        neg_gt = np.zeros((len(neg), 256, 256), dtype=np.bool)
        gt_pixel = np.concatenate((gt, neg_gt), 0)
        gt_image = np.concatenate((np.ones(pos.shape[0], dtype=np.bool), np.zeros(neg.shape[0], dtype=np.bool)), 0)

        pro_eva = evaluate(gt_pixel, scores, metric='pro')
        auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        print('Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(category, auc_pixel, auc_image_max, pro_eva))



def test(teacher, student, loader):
    teacher.eval()
    student.eval()
    loss_map = np.zeros((len(loader.dataset), 64, 64))
    i = 0
    for batch_data in loader:
        _, batch_img = batch_data
        batch_img = batch_img.cuda()
        with torch.no_grad():
            t_feat = teacher(batch_img)
            s_feat = student(batch_img)
        score_map = 1.
        m_fea = 0
        for j in range(len(t_feat)):
            # --------- M_fea ---------------
            cos_sim = 1 - torch.sum(t_feat[j] * s_feat[j], dim=1)/(torch.norm(t_feat[j], p=2, dim=1) * torch.norm(s_feat[j], p=2, dim=1)).unsqueeze(0)
            sm = cos_sim
            sm = sm.cuda()
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            m_fea = m_fea + sm
            # -------------------------------
        score_map = m_fea
        loss_map[i: i + batch_img.size(0)] = score_map.squeeze().cpu().data.numpy()
        i += batch_img.size(0)
    return loss_map

# Unique t_shapes
def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes

def train_val(teacher, student, train_loader, val_loader, args):
    min_err = np.inf

    teacher.eval()
    student.train()

    optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    for epoch in range(args.epochs):
        student.train()
        for batch_data in train_loader:
            _, batch_img = batch_data
            batch_img = batch_img.cuda()

            with torch.no_grad():
                t_feat = teacher(batch_img)
            s_feat = student(batch_img)

            loss = calculate_cd_spatial(t_feat, s_feat)

            print('[%d/%d] loss: %f' % (epoch, args.epochs, loss.item()))
            with open('loss.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                # Check if the file is empty
                if file.tell() == 0:
                    # Write headers if the file is empty
                    writer.writerow(['Epoch', 'Loss'])
                # Write epoch and loss values
                writer.writerow([epoch, loss.item()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        err = test(teacher, student, val_loader).mean()
        print('Valid Loss: {:.7f}'.format(err.item()))
        with open('val_loss.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            # Check if the file is empty
            if file.tell() == 0:
                # Write headers if the file is empty
                writer.writerow(['Epoch', 'Loss'])
            # Write epoch and loss values
            writer.writerow([epoch, err.item()])
        if err < min_err:
            min_err = err
            save_name = os.path.join(args.model_save_path, '15_class', 'best.pth.tar')
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            state_dict = {
                'state_dict': student.state_dict(),
                'epoch': epoch,
                'iter': -1
            }
            state_dict = student.state_dict()
            torch.save(state_dict, save_name)

if __name__ == "__main__":
    main()

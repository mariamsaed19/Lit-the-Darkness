import os
import Myloss
import dataloader
from modeling import model as model
import torch.optim
from modeling.fpn import *
from option import *
from utils import *
import torch.nn.functional as F
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display # to display images
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU only
device = get_device()
g_kernel_size = 5
g_padding = 2
sigma = 3
kx = cv2.getGaussianKernel(g_kernel_size,sigma)
ky = cv2.getGaussianKernel(g_kernel_size,sigma)
gaussian_kernel = np.multiply(kx,np.transpose(ky))
gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).to(device)
def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    gradient2_h = (img[:,:,4:,:]-img[:,:,:height-4,:]).abs()
    gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
    gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    return gradient_h*gradient2_h, gradient_w*gradient2_w

def illumination_smooth_loss(image, illumination):
    gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
    max_rgb, _ = torch.max(image, 1)
    max_rgb = max_rgb.unsqueeze(1)
    gradient_gray_h, gradient_gray_w = gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    weight_h = 1/(F.conv2d(gradient_gray_h, weight=gaussian_kernel, padding=g_padding)+0.0001)
    weight_w = 1/(F.conv2d(gradient_gray_w, weight=gaussian_kernel, padding=g_padding)+0.0001)
    weight_h.detach()
    weight_w.detach()
    loss_h = weight_h * gradient_illu_h
    loss_w = weight_w * gradient_illu_w
    max_rgb.detach()
    f = 1e7
    return (loss_h.sum() + loss_w.sum() + torch.norm(illumination-max_rgb, 1))/f

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()
        #self.eps = 1e-6
        self.eps = 0

    def forward(self, org, x ):

        #b,c,h,w = x.shape       
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mean_org = torch.mean(org,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        omr,omg, omb = torch.split(mean_org, 1, dim=1)
        
        Drg = torch.pow(omr-omg,2)-torch.pow(mr-mg,2)
        Drb = torch.pow(omr-omb,2)-torch.pow(mr-mb,2)
        Dgb = torch.pow(omb-omg,2)-torch.pow(mb-mg,2)
        '''
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        '''
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2) + self.eps,0.5)

        return k
class Trainer():
    def __init__(self):
        self.scale_factor = args.scale_factor
        self.net = model.enhance_net_nopool(self.scale_factor, conv_type=args.conv_type).to(device)
        self.seg = fpn(args.num_of_SegClass).to(device)
        self.seg_criterion = FocalLoss(gamma=2).to(device)
        self.train_dataset = dataloader.lowlight_loader(args.lowlight_images_path)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=args.train_batch_size,
                                                       shuffle=True,
                                                       num_workers=args.num_workers,
                                                       pin_memory=True)
        self.L_color = L_color()
        self.L_spa = Myloss.L_spa8(patch_size=args.patch_size)
        self.L_exp = Myloss.L_exp(8)
        self.L_TV = Myloss.L_TV()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.num_epochs = args.num_epochs
        self.E = args.exp_level
        self.grad_clip_norm = args.grad_clip_norm
        self.display_iter = args.display_iter
        self.snapshot_iter = args.snapshot_iter
        self.snapshots_folder = args.snapshots_folder
        if args.load_pretrain == True:
            self.net.load_state_dict(torch.load(args.pretrain_dir, map_location=device))
            print("weight is OK")


    def get_seg_loss(self, enhanced_image):
        # segment the enhanced image
        seg_input = enhanced_image.to(device)
        seg_output = self.seg(seg_input).to(device)

        # build seg output
        target = (get_NoGT_target(seg_output)).data.to(device)

        # calculate seg. loss
        seg_loss = self.seg_criterion(seg_output, target)

        return seg_loss


    def get_loss(self, A, enhanced_image, img_lowlight, E):
        Loss_TV = 1600 * self.L_TV(A)
        loss_spa = torch.mean(self.L_spa(enhanced_image, img_lowlight))
        loss_col = 5 * torch.mean(self.L_color(img_lowlight,enhanced_image))
        loss_exp = torch.mean(self.L_exp(enhanced_image, E))
        w = 10
        loss_seg = self.get_seg_loss(enhanced_image)
        loss_new = illumination_smooth_loss(img_lowlight, A)
        #print("TV:",Loss_TV,"\tspa:",loss_spa,"\trgb:",loss_col,"\texp:",loss_exp,"\tseg:",loss_seg)
        loss = Loss_TV + loss_spa + loss_col+ 0.25*loss_new + w*loss_exp + 0.1 * loss_seg

        return loss


    def train(self):
        self.net.train()
        print("start Training network")

        print("************")
        for epoch in range(self.num_epochs):
            print("\nStart epoch",epoch)
            for iteration, img_lowlight in enumerate(self.train_loader):
                img_lowlight = img_lowlight.to(device)
                transform = T.ToPILImage()
                '''
                img = transform(img_lowlight[0])
                display(img)
                '''
                enhanced_image, A = self.net(img_lowlight)
                loss = self.get_loss(A, enhanced_image, img_lowlight, self.E)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.net.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                if ((iteration + 1) % self.display_iter) == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())
                    if epoch<0:
                      f = plt.figure()
                      f.add_subplot(1,2, 1)
                      plt.imshow(img_lowlight[0].squeeze(0).permute(1, 2, 0).to('cpu').numpy())
                      f.add_subplot(1,2, 2)
                      plt.imshow(enhanced_image[0].squeeze(0).permute(1, 2, 0).to('cpu').detach().numpy())
                      plt.show(block=True)
                    
                    '''
                    plt.imshow(img_lowlight[0].squeeze(0).permute(1, 2, 0).to('cpu').numpy())
                    plt.show()
                    print(" enhanced image")
                    #plt.figure(figsize = (8,8))
                    plt.imshow(enhanced_image[0].squeeze(0).permute(1, 2, 0).to('cpu').detach().numpy())
                    
                    plt.show()
                    '''
                    
                if ((iteration + 1) % self.snapshot_iter) == 0:
                    torch.save(self.net.state_dict(), self.snapshots_folder + "Epoch" + str(epoch) + '.pth')



if __name__ == "__main__":
    t = Trainer()
    t.train()










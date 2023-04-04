# -*- coding=utf-8 -*-


import numpy as np
import torch
import time
import os
import torch.optim as optim
from PIL import Image
import torchvision.transforms.functional as TF
import openslide
import cv2
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from net.pytorch_net import PytorchNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors = [(plt.cm.jet(i)[0],plt.cm.jet(i)[1],plt.cm.jet(i)[2],i/255) for i in range(0,256)]
new_map = LinearSegmentedColormap.from_list('new_map', colors, N=256)

class Option(object):
    def __init__(self):
        self.img_path = '/disk2/mz_data/TCGA_GIST/543e1c89-a892-4ee8-8716-9c94136036a9/TCGA-IS-A3K6-01Z-00-DX1.C3EACA3B-0E4B-4019-BE7C-BD18373EE4F2.svs' # the image path waiting for inference
        self.model_dir = './model/' # the dir path you save the pre-trained model "STT_BOX_v1.pth"   
        self.save_TCGA_path = './'# the dir path for saving outputs
        self.seed = 1236
        self.crop_size = 512 
        self.input_size = 512 
        self.stride = 256
        self.class_num = 3 
        self.batchsize = 32

class STT_BOX(object):
    def __init__(self,params):
        self.params = params 
        torch.manual_seed(self.params.seed)
        torch.cuda.manual_seed(self.params.seed)
        print ("{}Initializing{}".format('='*10,'='*10))
        self._load_mynet_()
    def _load_mynet_(self):
        self.mynet = PytorchNet('resnet50',True,self.params.class_num).to(device)
        self.mynet.load_state_dict(torch.load(os.path.join(self.params.model_dir,'STT_BOX_v1.pth')))
    def __call__(self):
        t_start = time.time()
        self._test_TCGA_()
        print("TEST TIME: {}".format(time.time()-t_start))
    def _test_TCGA_(self):
        my_tiff = [self.params.img_path[self.params.img_path.rfind('/')+1:-4],(0,0),(0,0),self.params.stride]
        fore_ratio_frac = float(self.params.crop_size*self.params.crop_size) 
        print(my_tiff[0])
        time_tiff = time.time()
        slide = openslide.OpenSlide(self.params.img_path)
        my_tiff[2] = slide.level_dimensions[0]
        single_stride = my_tiff[3]
        ROI_w, ROI_h = my_tiff[2] 
        print(my_tiff[2])
        thumb_file = slide.get_thumbnail((ROI_w/200, ROI_h/200))
        thumb_file.save(os.path.join(self.params.save_TCGA_path,my_tiff[0]+"_thumb.png"), "png")
        result_arr_list = [] 
        pos_h = 0 
        region_height = 5120#10240#2048000#10240#10240 
        while pos_h<ROI_h:
            region_w = ROI_w
            if pos_h+region_height < ROI_h:
                region_h = region_height     
            else: 
                region_h = ROI_h - pos_h
            slide_ROI = slide.read_region((0,pos_h),0,(region_w,region_h)).convert("RGB")
            slide_ROI_grey = np.array(slide_ROI.convert('L'))
            slide_ROI_grey = (slide_ROI_grey>204).astype(np.int64)
            slide_ROI = np.array(slide_ROI).astype(np.float32)
            w_index = int(np.floor((region_w - self.params.crop_size)/single_stride)+1)
            h_index = int(np.floor((region_h - self.params.crop_size)/single_stride)+1)
            if ((w_index>0) and (h_index>0)):
                result_arr = np.zeros((h_index, w_index, 4)).astype(np.float32) 
                for h in range(h_index):
                    print("{}/{},{}/{}".format(pos_h,ROI_h,h,h_index))
                    batch_input = np.zeros((1, self.params.crop_size, self.params.crop_size, 3)).astype(np.float32) # 裁剪patch暂存
                    w_pos = 0 
                    fore_tag = 0 
                    for w in range(w_index):
                        patch = slide_ROI[h*single_stride : h*single_stride+self.params.crop_size,
                                            w*single_stride : w*single_stride+self.params.crop_size,:]
                        patch_fore_ratio = 1-slide_ROI_grey[h*single_stride : h*single_stride+self.params.crop_size,
                                            w*single_stride : w*single_stride+self.params.crop_size].sum()/fore_ratio_frac
                        if patch_fore_ratio<0.1:
                            result_arr[h][w][3] = 0
                        else:
                            result_arr[h][w][3] = patch_fore_ratio 
                            fore_tag = 1
                        patch = patch[np.newaxis,:,:,:]
                        batch_input = np.concatenate((batch_input, patch), axis=0)
                        if batch_input.shape[0]>self.params.batchsize:
                            if fore_tag==0:
                                batch_temp = batch_input[1:]
                            else:
                                batch_temp = self._single_batch_(batch_input[1:])
                                result_arr[h,w_pos:w_pos+batch_temp.shape[0],:3] = batch_temp 
                            w_pos = w_pos+batch_temp.shape[0]
                            batch_input = np.zeros((1, self.params.crop_size, self.params.crop_size, 3)).astype(np.float32) 
                            fore_tag=0
                    if batch_input.shape[0]>1:
                        if fore_tag==0:
                            batch_temp = batch_input[1:]
                        else:
                            batch_temp = self._single_batch_(batch_input[1:])
                            result_arr[h,w_pos:w_pos+batch_temp.shape[0],:3] = batch_temp 
                result_arr_list.append(result_arr) 
            pos_h = pos_h + region_height-256 
        region_num = len(result_arr_list)
        if region_num==1:
            result_arr_final = result_arr_list[0]
        else:
            result_arr_final = result_arr_list[0]
            for i in range(1,region_num):
                result_arr_final = np.concatenate((result_arr_final,result_arr_list[i]),axis=0)

        show_temp = result_arr_final[:,:,0].copy() 
        result_arr_final = self.get_GIST_fore(result_arr_final)
        mask = result_arr_final[:,:,0].copy()
        mask = self.remove_small_points(show_temp, mask,200,0.5)
        show_temp = show_temp*mask 
        show_temp = cv2.blur(show_temp,(3,3))
        saveimg = plt.imshow(show_temp, cmap = new_map,vmin=0,vmax=1)
        plt.axis("off")
        plt.savefig("./"+my_tiff[0]+"_heatmap.png",transparent = True,bbox_inches='tight',pad_inches=0.0)
        plt.close()
        print("{}:{}s".format(my_tiff[0],time.time()-time_tiff))
        slide.close()

    def _single_batch_(self, batch):
        self.mynet.eval()
        batch = self._normalize_(batch)
        with torch.no_grad():
            output_batch_cla = self.mynet(batch)
            output_batch_cla = torch.softmax(output_batch_cla, dim=1)
        output_batch_cla = output_batch_cla.cpu().data.numpy()
        return output_batch_cla
    def _normalize_(self, batch):
        batch = np.array(batch).astype(np.float32)
        batch = batch.transpose((0, 3, 1, 2))
        batch = torch.from_numpy(batch).float().to(device)
        return batch
    def my_swap(self, r_temp): 
        GIST_grey = (r_temp.copy()*255).astype(np.uint8)
        GIST_grey = cv2.dilate(GIST_grey, np.ones((7,7),np.uint8), iterations=1)
        GIST_grey = cv2.erode(GIST_grey, np.ones((7,7),np.uint8), iterations=1)
        GIST_grey = GIST_grey.astype(np.float16)
        r_temp[GIST_grey>0] = 1
        return r_temp
    def get_GIST_fore(self, result_arr):
        result_arr[:,:,3][result_arr[:,:,3]<0.2] = 0 
        result_arr[:,:,3][result_arr[:,:,3]>0] = 1 
        result_arr[:,:,0][result_arr[:,:,0]<0.3] = 0 
        result_arr[:,:,0][result_arr[:,:,0]>0] = 1 
        result_arr[:,:,0] = result_arr[:,:,0]*result_arr[:,:,3] 
        result_arr[:,:,0] = self.my_swap(result_arr[:,:,0])*result_arr[:,:,3]
        return result_arr 
    def remove_small_points(self, show_temp,image,area_threshold,pred_threshold):
        img_label, num = measure.label(image, connectivity=2, return_num=True)
        props = measure.regionprops(img_label)
        resMatrix = np.zeros(img_label.shape)
        for i in range(0, len(props)):
            if props[i].area > area_threshold:
                tmp = (img_label == i + 1).astype(np.float32)
                if (show_temp*tmp).sum() > pred_threshold*tmp.sum():
                    tmp = tmp.astype(np.uint8)
                    resMatrix += tmp 
        return resMatrix
if __name__ == "__main__":
    opt = Option()
    obj = STT_BOX(opt)
    obj()


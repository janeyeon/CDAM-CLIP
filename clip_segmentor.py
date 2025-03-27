import torch
import torch.nn as nn
import sys 
sys.path.append("..")

import clip
from prompts.imagenet_template import openai_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS

from pamr import PAMR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os


import matplotlib.pyplot as plt


def entropy_sim(probabilities):
    entropy = -torch.sum(probabilities * torch.log2(probabilities), axis=0) # [28, 28]

    return entropy

@MODELS.register_module()
class CLIPForSegmentation(BaseSegmentor):
    def __init__(self, clip_path, name_path, device=torch.device('cuda'),
                    pamr_steps=0, pamr_stride=(8,16), prob_thd=0.0, logit_scale=40, 
                    slide_stride=224, slide_crop=448, area_thd=None):
        
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True)
        super().__init__(data_preprocessor=data_preprocessor)
        self.net, _ = clip.load(clip_path, device=device, jit=False)
        
        num = 80
        
        
        
        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words) -num
        self.num_classes = max(self.query_idx) + 1 -num 
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        self.query_idx = self.query_idx[:-num ] 
        



        query_features = []
        self.query_words = query_words

        with torch.no_grad():
            for qw in query_words:
                query = clip.tokenize([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.net.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
                

        
        self.query_features_ori = torch.cat(query_features, dim=0)
        self.query_features = self.query_features_ori[:-num,:]
        
        
        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.area_thd = area_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.align_corners = False
        self.device = device
        

        
       # pamr_steps = 1
        if pamr_steps > 0:
            self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device)
        else:
            self.pamr = None
            
        self.slide = 0
        self.window = 0
        self.m = nn.Softmax(dim=0)

        self.kl_temp_weight = torch.zeros(12,785,785, dtype=torch.float16).to(self.device) 
        
    
        self.total_segments = []
    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]

        image_features = self.net.encode_image(img, self.window, self.slide, return_all=True, csa=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        image_features = image_features[:, 1:]
        
        logits = image_features @ self.query_features_ori.T
        patch_size = self.net.visual.patch_size
        w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
        out_dim = logits.shape[-1]
        
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        logits_no_inter = logits.clone()

        with torch.no_grad():
            
            kl_temp_weight = self.kl_temp_weight 
           
            T = 0.1 # 0.1 # 0.2 for voc
            P = 0.1 #[0.1,0.1,0.1,0.1] # 0.1
                
            
                
            logits_sm = self.m(logits_no_inter[0,:,:,:]/T)
            

            c, w_attn, h_attn = logits_sm.shape
            if w_attn*h_attn +1 != 785:
                kl_temp_weight = torch.zeros(12,w_attn*h_attn + 1, w_attn*h_attn + 1, dtype=torch.float16).to(self.device) 
                
                
            kl_sizes = [0.25, 0.37, 0.5, 0.63, 0.75, 0.87]
            
            kl_size_w = [int(w_attn*size) for size in kl_sizes]
            kl_size_h = [int(h_attn*size) for size in kl_sizes]
            

            logits_flatten = logits_sm.reshape(c,-1) # 50 196
            ######

            #original size
            softmax_temp = nn.Softmax(dim=1)
            p_temp = logits_flatten.unsqueeze(2).expand(c, w_attn * h_attn, w_attn * h_attn)
            q_temp = logits_flatten.unsqueeze(1).expand(c, w_attn * h_attn, w_attn * h_attn)
            M = (p_temp+q_temp) * 0.5
            
            #jhonson
            kl_temp =0.5 *(torch.sum(logits_flatten.unsqueeze(2) *  (torch.log(p_temp + 1e-8) - torch.log(M + 1e-8)), dim=0) + torch.sum(logits_flatten.unsqueeze(1) *  (torch.log(q_temp + 1e-8) - torch.log(M + 1e-8)), dim=0))


            kl_temp = 1 - kl_temp


            min_vals = kl_temp.min(dim=-1, keepdim=True)[0].expand( w_attn * h_attn, w_attn * h_attn)  # Find minimum values along dim=2, keep dimensions
            max_vals = kl_temp.max(dim=-1, keepdim=True)[0].expand( w_attn * h_attn, w_attn * h_attn)  # Find maximum values along dim=2, keep dimensions

            kl_temp = (kl_temp - min_vals) / (max_vals - min_vals + 1e-8)
            kl_temp_ori_soft = softmax_temp(kl_temp/P) 
            

            min_vals = kl_temp_ori_soft.min(dim=-1, keepdim=True)[0].expand( w_attn * h_attn, w_attn * h_attn)  # Find minimum values along dim=2, keep dimensions
            max_vals = kl_temp_ori_soft.max(dim=-1, keepdim=True)[0].expand( w_attn * h_attn, w_attn * h_attn)  # Find maximum values along dim=2, keep dimensions
            kl_temp_ori = (kl_temp_ori_soft - min_vals) / (max_vals - min_vals + 1e-8)
        
            logits_no_inter_clone = logits_no_inter.clone()

            # multi-scale version
            step = 0
            attn_list = []
            for kl_size_w_temp, kl_size_h_temp in zip(kl_size_w, kl_size_h):
                logits_no_inter = nn.functional.interpolate(logits_no_inter_clone, size=(kl_size_w_temp,kl_size_h_temp), mode='bilinear')

                logits_sm = self.m(logits_no_inter[0,:,:,:]/T)

                c, w_attn, h_attn = logits_sm.shape

                logits_flatten = logits_sm.reshape(c,-1) # 50 


                #fast version
                softmax_temp = nn.Softmax(dim=1)
                p_temp = logits_flatten.unsqueeze(2).expand(c, w_attn * h_attn, w_attn * h_attn)
                q_temp = logits_flatten.unsqueeze(1).expand(c, w_attn * h_attn, w_attn * h_attn)
                M = (p_temp+q_temp) * 0.5



                #jhonson
                kl_temp =0.5 *(torch.sum(logits_flatten.unsqueeze(2) *  (torch.log(p_temp + 1e-8) - torch.log(M + 1e-8)), dim=0) + torch.sum(logits_flatten.unsqueeze(1) *  (torch.log(q_temp + 1e-8) - torch.log(M + 1e-8)), dim=0))

                ####
                kl_temp = nn.functional.interpolate(kl_temp.unsqueeze(0).reshape(1,w_attn * h_attn, w_attn , h_attn), size=(w,h), mode='bilinear') #1 100 14 14
                kl_temp = kl_temp.squeeze().permute(1,2,0).reshape(w,h,w_attn , h_attn) # 100 14 14 -> 14 14 10 10
                kl_temp = nn.functional.interpolate(kl_temp, size=(w,h), mode='bilinear') # 14 14 14 14
                kl_temp = kl_temp.permute(2,3,0,1).reshape(w*h,w , h).reshape(w*h,w*h)
                w_attn = w
                h_attn = h

                ####

                kl_temp = 1 - kl_temp


                min_vals = kl_temp.min(dim=-1, keepdim=True)[0].expand( w_attn * h_attn, w_attn * h_attn)  # Find minimum values along dim=2, keep dimensions
                max_vals = kl_temp.max(dim=-1, keepdim=True)[0].expand( w_attn * h_attn, w_attn * h_attn)  # Find maximum values along dim=2, keep dimensions

                kl_temp = (kl_temp - min_vals) / (max_vals - min_vals + 1e-8)
                kl_temp_1_soft = softmax_temp(kl_temp/P) 

            #    if i == 0:
                min_vals = kl_temp_1_soft.min(dim=-1, keepdim=True)[0].expand( w_attn * h_attn, w_attn * h_attn)  # Find minimum values along dim=2, keep dimensions
                max_vals = kl_temp_1_soft.max(dim=-1, keepdim=True)[0].expand( w_attn * h_attn, w_attn * h_attn)  # Find maximum values along dim=2, keep dimensions
                kl_temp_1 = (kl_temp_1_soft - min_vals) / (max_vals - min_vals  + 1e-8)
                
                
                attn_list.append(kl_temp_1)

                

            kl_temp_weight[:, 1:, 1:] =(torch.stack(attn_list, dim=0).sum(0) +1.0*kl_temp_ori ) / (len(kl_size_w) + 1.0)  
            kl_temp_weight[:,:,0] = 0.0    




            ##re-prediction
            x = self.net.visual.intermediate_feat_before_last
            for blk in self.net.visual.transformer.resblocks[-1:]:
                x = x + self.net.visual.custom_dual_attn(blk.attn, blk.ln_1(x), weight = kl_temp_weight, csa=True)
                x = x + blk.mlp(blk.ln_2(x))
            self.net.visual.intermediate_feat_after_last = x

            x = x.permute(1, 0, 2)  # LND -> NLDd

            image_features_temp = self.net.visual.ln_post(x) @ self.net.visual.proj
            image_features_temp /= image_features_temp.norm(dim=-1, keepdim=True)
            image_features_temp = image_features_temp[:, 1:]

            image_features_temp_ori = image_features_temp.clone()
            logits = image_features_temp_ori @ self.query_features.T

            patch_size = self.net.visual.patch_size
            w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
            out_dim = logits.shape[-1]
            logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
            logits_no_inter = logits


        attr_logits = image_features_temp_ori @ self.query_features.T
        out_dim = attr_logits.shape[-1]
        attr_logits = attr_logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        
        
        probability = softmax_temp(attr_logits*50).squeeze() # [1, 21, 28, 28]
        entropy = entropy_sim(probability) #[28, 28]

        
        self.prob_thd = 0.5 *2.5/((entropy.min()+entropy.max())/2.0)
        
   
    
        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        ######

       
        return logits


    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        
        self.window = 0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.forward_feature(crop_img)
                preds += nn.functional.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
                
                self.window += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        if self.pamr:
            img = nn.functional.interpolate(img, size=img_size, mode='bilinear')
            logits = self.pamr(img, logits.to(img.dtype)).to(self.dtype)

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
            
        self.slide += 1
        
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)
    
    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0) # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            if self.area_thd is not None:
                # Force segmentations with area < self.area_thd to 0 (background)
                predictions = nn.functional.one_hot(seg_logits.argmax(0), num_cls).to(seg_logits.dtype)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)  # prone background
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits.dtype)          
                seg_logits[1:] *= area_pred.transpose(0, -1)
            
            seg_pred = seg_logits.argmax(0, keepdim=True)
            
            if self.prob_thd is not None: 
                seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0
            
            
            
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': seg_pred})
            })

        return data_samples
    
    def _forward(data_samples):
        """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
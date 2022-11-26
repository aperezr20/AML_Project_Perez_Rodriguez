import traceback
import numpy as np
import pycocotools.mask as m
import skimage.io as io
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import os
import gc
from matplotlib.colors import Normalize
from copy import copy, deepcopy
import json 
import argparse

def roundbox(box):
    keys = box.split(' ')
    keys = [str(round(float(k),4)) for k in keys]
    return ' '.join(keys)

def floatbox(box):
    keys = box.split(' ')
    return list(map(float,keys))

def boxiou(bb1,bb2):
    x1 = max(bb1[0],bb2[0])
    y1 = max(bb1[1],bb2[1])
    x2 = min(bb1[2],bb2[2])
    y2 = min(bb1[3],bb2[3])
    if x2<x1 or y2<y1:
        return 0.0
    inter = (x2-x1)*(y2-y1)
    area1 = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    if (area1+area2-inter)==0:
        breakpoint()
    box_iou = inter/(area1+area2-inter)

    assert box_iou>=0 and box_iou<=1
    return box_iou

def getrealbox(boxes,box):
    maxiou = 0
    realbox = ''
    box = floatbox(box)
    for box2 in boxes:
        iou = boxiou(floatbox(box2),box)
        if iou>maxiou:
            maxiou=iou
            realbox=box2
    
    assert maxiou>0.9 and maxiou<=1, 'Ugh ya ni se {}'.format(maxiou)
    return realbox
    
def NMS(instances):

    all_masks = []
    all_instances = []
    for iid,ins in enumerate(instances):
        this_mask = m.decode(ins['segmentation'])
        if len(all_masks)==0:
            all_masks.append(this_mask)
            all_instances.append(ins)
        else:
            max_iou = 0
            for mask in all_masks:
                intersection = np.sum(mask * this_mask)
                union = np.sum(mask) + np.sum(this_mask)
                iou = intersection/(union-intersection)
                if iou>max_iou:
                    max_iou=iou
                if iou>=0.9:
                    break
            
            if max_iou<0.9:
                all_masks.append(this_mask)
                all_instances.append(ins)
    
    return sorted(all_instances,key=lambda x: x['score'])

def eval_detection(preds,data,path):
    cats = {1,2,3,4,5,6,7}
    #try:
    file_dict ={}
    file_name2id = {}
    for d in data['images']:
        file_name = d['file_name']
        info = file_name.split('_')
        sequence = '_'.join(info[:2])
        file_name2id[d['id']] = file_name

        # if "seq_2" not in file_name:
        file_dict[file_name] = {'file_name': file_name, 'id': d['id'], 'instances': [], 'categories': [], 'video_name': sequence}

    data=None
    gc.collect()
    
    # Final
    # thresh = {1: 0.9, 2: 0.8, 3: 0.01, 4: 0.9, 5: 0.05, 6: 0.001, 7: 0.0001}
    # class_inds = {1: 3, 2: 3, 3: 3, 4: 3, 5: 2, 6: 2, 7: 1}

    # 5999
    # thresh = {1: 0.9, 2: 0.8, 3: 0.01, 4: 0.9, 5: 0.1, 6: 0.001, 7: 0.0001}
    # class_inds = {1: 3, 2: 3, 3: 3, 4: 3, 5: 1, 6: 1, 7: 1}
    
    # SwinS
    thresh = {1: 0.65, 2: 0.84, 3: 0.007, 4: 0.94, 5: 0.1, 6: 0.005, 7: 0.0006}
    class_inds = {1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1}

    # R50
    # thresh = {1: 0.9, 2: 0.9, 3: 0.001, 4: 0.9, 5: 0.5, 6: 0.001, 7: 0.001}
    # class_inds = {1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1}

    for pred in tqdm(preds,desc='predictions'):
        class_inss = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
        file_name = file_name2id[pred['image_id']]
        
        categories = []
        instances = []

        for ins in pred['instances']:
            if 'mask_embd' in ins:
                del ins['mask_embd']
            if 'decoder_out' in ins:
                del ins['decoder_out']
            if ins['score']>0.0:
                ins['category_id'] += 1
                # instances.append(ins)
                class_inss[ins['category_id']].append(ins)
        
        for cat in class_inss:
            this_list = class_inss[cat]
            this_list.sort(reverse=True,key= lambda x: x['score'])
            if len(this_list)>0:
                th = thresh[cat]
                max_ind = class_inds[cat]
                instances.extend([u for u in this_list[:max_ind] if u['score']>=th])            

        instances.sort(key=lambda x: x['score'])

        file_dict[file_name]['instances'] = instances
        file_dict[file_name]['categories'] = set(categories)

    preds = None
    instances = None
    gc.collect()

    ious = []
    gt_ious = []
    pcls_ious = {1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    for file in tqdm(file_dict):

        # if seq and 'seq_2' in file:
        #     continue

        gt_img = io.imread(os.path.join('/','media','SSD0','nayobi','All_datasets','coco_endovis_2018','annotations','instances_val2018',file))
        gt_img[gt_img==6]=4
        gt_img[gt_img==8]=5
        gt_img[gt_img==9]=6

        # gt_img[gt_img==6]=4
        # gt_img[gt_img==7]=5
        # gt_img[gt_img==8]=6
        # gt_img[gt_img==9]=7

        gt_classes = set(np.unique(gt_img))
        gt_classes.remove(0)

        # categories = file_dict[file]['categories']
        sem_im = np.zeros((1024,1280))
        for ins in file_dict[file]['instances']: #instances:
            p_mask = m.decode(ins['segmentation'])
            sem_im[p_mask==1]=ins['category_id']

        categories = set(np.unique(sem_im))
        categories.remove(0)

        # if categories2 != categories:
        #     breakpoint()

        class_iou = []
        gt_class_iou=[]
        for label in cats:
            if label in gt_classes or label in categories:
                pred_im = (sem_im==label).astype('uint8')
                gt_mask = (gt_img==label).astype('uint8')

                intersection = np.sum(pred_im * gt_mask)
                union = np.sum(pred_im) + np.sum(gt_mask)
                im_IoU = intersection/(union-intersection)
                assert im_IoU>=0 and im_IoU<=1, im_IoU
                class_iou.append(im_IoU)
                pcls_ious[label].append(im_IoU)

                if label in gt_classes and label not in categories:
                    assert im_IoU == 0
                
                if label not in gt_classes and label in categories:
                    assert im_IoU == 0

                if label in gt_classes:
                    gt_class_iou.append(im_IoU)

        assert len(class_iou)==len(gt_classes.union(categories))
        assert len(gt_class_iou)==len(gt_classes)

        if len(class_iou)>0:
            ious.append(float(np.mean(class_iou)))

        if len(gt_class_iou)>0:
            gt_ious.append(float(np.mean(gt_class_iou)))

    total_iou = float(np.mean(ious))
    total_gt_iou = float(np.mean(gt_ious))

    for cls in pcls_ious:
        pcls_ious[cls] = float(np.mean(pcls_ious[cls])) if len(pcls_ious[cls])>0 else np.nan
    total_ciou = float(np.mean(list(pcls_ious.values())))

    #json.dump({'cIoU':total_gt_iou,'IoU':total_iou,'cIoU':total_ciou,'class':pcls_ious},open(path,'w'))
    #except:
        #traceback.print_exc()
        #breakpoint()

    gc.collect()

    return total_gt_iou,total_iou,total_ciou,pcls_ious  

def visualize(preds,data,path):
    cats = {1,2,3,4,5,6,7}

    file_dict ={}
    file_name2id = {}
    for d in data['images']:
        file_name = d['file_name']
        info = file_name.split('_')
        sequence = '_'.join(info[:2])
        file_name2id[d['id']] = file_name

        file_dict[file_name] = {'file_name': file_name, 'id': d['id'], 'instances': [], 'categories': [], 'video_name': sequence}
    
    data=None
    gc.collect()
    
    # SwinS
    thresh = {1: 0.65, 2: 0.84, 3: 0.007, 4: 0.94, 5: 0.1, 6: 0.005, 7: 0.0006}
    class_inds = {1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1}

    for pred in tqdm(preds,desc='predictions'):
        class_inss = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
        file_name = file_name2id[pred['image_id']]

        categories = []
        instances = []

        for ins in pred['instances']:
            if 'mask_embd' in ins:
                del ins['mask_embd']
            if 'decoder_out' in ins:
                del ins['decoder_out']
            if ins['score']>0.0:
                ins['category_id'] += 1
                # instances.append(ins)
                class_inss[ins['category_id']].append(ins)
        
        for cat in class_inss:
            this_list = class_inss[cat]
            this_list.sort(reverse=True,key= lambda x: x['score'])
            if len(this_list)>0:
                th = thresh[cat]
                max_ind = class_inds[cat]
                instances.extend([u for u in this_list[:max_ind] if u['score']>=th])            

        instances.sort(key=lambda x: x['score'])

        file_dict[file_name]['instances'] = instances
        file_dict[file_name]['categories'] = set(categories)

    preds = None
    instances = None
    gc.collect()

    seq, frame = path.split(',')
    file = 'seq_{}_frame{}.png'.format(seq, frame)

    try:
        os.makedirs(os.path.join('Visualizations_RMATIS_Demo'),exist_ok=True)
        img = io.imread(os.path.join('/','media','SSD0','nayobi','All_datasets','coco_endovis_2018','val2018',file))
        gt_img = io.imread(os.path.join('/','media','SSD0','nayobi','All_datasets','coco_endovis_2018','annotations','instances_val2018',file)).astype('float')
    except:
        assert False, 'File ' + file + ' does not exist, try another. Sequence must exist and the frame number must have 3 digits (i.e: for frame 1 use 001).'

    gt_img[gt_img==6]=4
    gt_img[gt_img==8]=5
    gt_img[gt_img==9]=6

    gt_classes = set(np.unique(gt_img))
    gt_classes.remove(0)

    sem_im = np.zeros((1024,1280))
    for ins in file_dict[file]['instances']: #instances:
        p_mask = m.decode(ins['segmentation'])
        sem_im[p_mask==1]=ins['category_id']

    fig, axs = plt.subplots(1, 3, constrained_layout=True)

    cmap = plt.cm.Set1.copy()
    cmap.set_bad(color='black')

    gt_img[gt_img==0] = 'nan'

    sem_im = sem_im.astype('float')
    sem_im[sem_im==0] = 'nan'

    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    axs[1].imshow(sem_im, cmap=cmap,norm=Normalize(1,7))
    axs[1].axis('off')
    axs[1].set_title('RMATIS')

    axs[2].imshow(gt_img,cmap=cmap,norm=Normalize(1,7))
    axs[2].axis('off')
    axs[2].set_title('GroundTruth')
    
    plt.axis('off')

    plt.savefig(os.path.join('Visualizations_RMATIS_Demo', file))
    plt.close()        

    gc.collect()


    gc.collect()
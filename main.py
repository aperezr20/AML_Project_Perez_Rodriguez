import argparse
import os
import torch
import glob
import os
import json
from metrics import eval_detection

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--mode', help='test or demo mode', type=str,default='test')
    parser.add_argument('--img', help='the test image on which to calculate the results', type=str, default='')
    args = parser.parse_args()

    if args.mode == 'test':
        m_path = os.path.join('/','media','SSD0','aperezr20','endovis','Mask2Former','output','EndoVis_2018','Delta_4')

        preds = torch.load(os.path.join(m_path,'inference','instances_predictions.pth'))
        data = json.load(open(os.path.join('/','media','SSD0','nayobi','All_datasets','coco_endovis_2018','annotations','instances_val2018_transformed.json')))
        path = os.path.join('semantic_ious.json')

        metrics = eval_detection(preds,data,path)
        #breakpoint()
        print('--------------- Overall metrics on the validation split -------------------')
        print('mIoU: {0:.4f}   IoU: {0:.4f}   mcIoU: {0:.4f}   '.format(metrics[0],metrics[1],metrics[2]))
        print('--------------- Metrics per class -------------')
        print('BF: {0:.4f}   PF: {0:.4f}   LND: {0:.4f}   VS/SI: {0:.4f}   \n GR/CA: {0:.4f}   MCS: {0:.4f}   UP: {0:.4f}   '.format(metrics[3][1],metrics[3][2],metrics[3][3],metrics[3][5],metrics[3][6],metrics[3][4],metrics[3][7]))
        
    
    elif args.mode == 'demo':
        print('------- Evaluating the best model on the image {} --------'.format(args.img))


    
if __name__ == '__main__':
    main()
from gluoncv import model_zoo, data, utils, mx
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse

def resize_bb(bb, img_shape, min_size=32):
    w = bb[2] - bb[0]
    h = bb[3] - bb[1]
    if w < h:
        l = h - w
        bb[0] -= (l//2 + 1)
        bb[2] += l//2 + 1
        if w + l < img_shape[1]:
            if bb[0] < 0:
                bb[2] -= bb[0]
                bb[0] = 0
                    
            if bb[2] > img_shape[1]:
                bb[0] -= (bb[2] - img_shape[1])
                bb[2] = img_shape[1] - 1
        else:
            bb[0] = 0
            bb[2] = img_shape[1] - 1
            
    if h < w:
        l = w - h
        
        bb[1] -= (l//2 + 1)
        bb[3] += l//2
        if h + l < img_shape[0]:
            if bb[1] < 0:
                bb[3] -= bb[1]
                bb[1] = 0
                    
            if bb[3] > img_shape[0]:
                bb[1] -= (bb[3] - img_shape[0])
                bb[3] = img_shape[0] - 1
        else:
            bb[1] = 0
            bb[3] = img_shape[0] - 1
    
    bb[0] = max(bb[0], 0)
    bb[1] = max(bb[1], 0)
    bb[2] = min(bb[2], img_shape[1] - 1)
    bb[3] = min(bb[3], img_shape[0] - 1)
    
    return bb


def crop(imgs_path, crop_path, net):
    try:
        os.mkdir(crop_path)
    except:
        pass
    bird = [i for i in range(len(net.classes)) if net.classes[i] == 'bird'][0]
    
    for fname in tqdm(os.listdir(imgs_path)):
        try:
            os.mkdir(crop_path + '/' +  fname)
        except:
            pass

        for img_name in os.listdir(imgs_path + '/' + fname):
            x, img = data.transforms.presets.ssd.load_test(imgs_path + '/' + fname + '/' + img_name, short=512)
            class_IDs, scores, bounding_boxes = net(x)
            class_IDs, scores, bounding_boxes = class_IDs.asnumpy()[0, :, 0], scores.asnumpy()[0, :, 0], bounding_boxes.asnumpy()[0, :]
            class_IDs = class_IDs[np.argwhere(scores > 0.5)]
            bounding_boxes = bounding_boxes[np.argwhere(scores > 0.5)]

            if bird in class_IDs:
                bb = bounding_boxes[class_IDs == bird][0]
                bb = np.array(bb, dtype=int)
                bb = resize_bb(bb, img.shape[:2])
                img = img[bb[1]:bb[3], bb[0]:bb[2], :]

            plt.imsave(crop_path + '/' + fname + '/' + img_name, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect bounding boxes over images and crop them')
    parser.add_argument('--data_dir', type=str,
                        help="folder where images are located. ")
    parser.add_argument('--output_dir', type=str,
                        help="folder where results are stored ")
    parser.add_argument('--model', type=str, metavar='M', default='ssd_512_resnet50_v1_voc',
                        help="model name for object detection")

    args = parser.parse_args()

    # Load the model 
    net = model_zoo.get_model(args.model, pretrained=True)

    crop(args.data_dir, args.output_dir, net)
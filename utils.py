import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch 
import numpy as np
import cv2
import PIL
import torch.nn.functional as F
from tqdm import tqdm


def extract_bbox(img) :
    ori_img = img.copy()
    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(img,100,255,0)
    contours, _  = cv2.findContours(np.uint8(thresh),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        x,y,w,h = cv2.boundingRect(biggest_contour)
        return ori_img[y:y+h,x:x+w,:]
    else:
        return ori_img
def standalize_image( img_inp , do_extract = True, resize_shape = 224) :
    back_ground_img = np.zeros(shape=(resize_shape,resize_shape,3)) 
    if do_extract :
        img_inp = extract_bbox(img_inp)
        inp_resize_shape = 140 #np.random.choice(np.arange(120, 200, 20))
    else:
        inp_resize_shape = 224
    h,w,_ = img_inp.shape
    h_resize, w_resize = int(inp_resize_shape/max(h,w) * h), int(inp_resize_shape/max(h,w) * w)
    img_resize = cv2.resize(img_inp,(w_resize,h_resize))
    x_start_ind = (resize_shape - w_resize) // 2
    y_start_ind = (resize_shape - h_resize) // 2
    back_ground_img[y_start_ind:(y_start_ind + h_resize), x_start_ind : (x_start_ind + w_resize),:] = img_resize
    return np.uint8(back_ground_img)

def cv2_to_torch(img, transform,  do_stand = True, do_extract = False) :
    if do_stand :
        img = standalize_image(img, do_extract)
    standalized_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(np.uint8(img))
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    return img,standalized_img

class ShapeRecogModel() :
    def __init__(self, model_path, querry_with_backbone = False) :
        self.model = self.load_model(model_path= model_path, querry_with_backbone = querry_with_backbone)
        self.querry_with_backbone = querry_with_backbone
        self.transform  = transforms.Compose([            
    # transforms.Resize((resize_shape,resize_shape)),              
            transforms.ToTensor(),             
            transforms.Normalize(                      
            mean=[0.485, 0.456, 0.406],                
            std=[0.229, 0.224, 0.225]                  
        )])      
    def load_model(self, model_path, querry_with_backbone = False) :
        print('LOADING MODEL !!')
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        checkpoint = torch.load(model_path)

        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') :
                if not querry_with_backbone:
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                elif not k.startswith('module.encoder_q.fc'):
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]
        resnet50 = models.resnet50(pretrained = True)
        if not querry_with_backbone :
            resnet50.fc = nn.Linear(2048,128)
            resnet50.fc = nn.Sequential(nn.Linear(2048, 2048),nn.ReLU(),resnet50.fc)
        msg = resnet50.load_state_dict(state_dict, strict=False)
        print(msg)
        if querry_with_backbone :
            modules =  list(resnet50.children())[:-1]
            resnet50 = nn.Sequential(*modules)
        resnet50 = resnet50.cuda()
        print('DONE LOADING')
        return resnet50

    def extract_feature(self, image = None, image_path = None) :
        self.model.eval()
        assert image is not None or image_path is not None
        if image_path is not None :
            image = cv2.imread(image_path)
        image, _ = cv2_to_torch(image,self.transform, do_extract = True)
        image = image.cuda()
        with torch.no_grad():
            gal_img_embd = self.model(image)[0,:,0,0].cpu().data.numpy() if self.querry_with_backbone \
                else self.model(image)[0].cpu().data.numpy()
        return gal_img_embd

        
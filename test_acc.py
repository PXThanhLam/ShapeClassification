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

resize_shape = 224
def merge_im(im1, im2, label_1 = None, label_2 = None) :
    h1,w1,_ = im1.shape
    h2,w2,_ = im2.shape
    assert h1 == h2
    res = np.zeros((h1,w1 + w2,3))
    res[0:h1,0:w1,:] = im1
    res[0:h1,w1:w2+w1,:] = im2
    if label_1 != None :
        cv2.putText(res, label_1, (30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    if label_1 != None :
        cv2.putText(res, label_2, (w1+30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    return res

def cosine_sim(emb1, emb2) :
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1,emb2)
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
def standalize_image( img_inp , do_extract = True) :
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

transform = transforms.Compose([            
    # transforms.Resize((resize_shape,resize_shape)),              
    transforms.ToTensor(),             
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
 )])
def cv2_to_torch(img, do_stand = True, do_extract = False) :
    if do_stand :
        img = standalize_image(img, do_extract)
    standalized_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(np.uint8(img))
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    return img,standalized_img

print('!!! LOAD MODEL')
checkpoint = torch.load('/home/lam/CYBER/Factory/ShapeClassification/checkpoint_0005.pth.tar')
state_dict = checkpoint['state_dict']
state_dict_k = state_dict.copy()

querry_with_backbone = False
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
resnet50.eval()
print('DONE LOADING')


test_root = '/home/lam/CYBER/Factory/ShapeClassification/img/fake_gen_fabric_shape'
query_embds = {}
query_std_imgs = {}
for query_folder in tqdm(os.listdir(test_root)):
    for query_path in os.listdir(test_root + '/' + query_folder) :
        if 'mask' not in query_path:
            continue
        test_path = test_root +  '/' + query_folder + '/' + query_path
        querry_img = cv2.imread(test_path)
        querry_img,query_std_img = cv2_to_torch(querry_img,do_extract = True)
        querry_img = querry_img.cuda()
        with torch.no_grad():
            querry_img_embd = resnet50(querry_img)[0,:,0,0].cpu().data.numpy() if querry_with_backbone else resnet50(querry_img)[0].cpu().data.numpy()
        query_embds[test_path] = querry_img_embd
        query_std_imgs[test_path] = query_std_img

gal_path = '/home/lam/CYBER/Factory/ShapeClassification/img/template_1'
gal_embds = {}
gal_std_imgs = {}
for gal_folder in tqdm(os.listdir(gal_path)) :
    for gal_img_path in os.listdir(gal_path + '/' + gal_folder) :
        if gal_img_path.endswith('.png') or  gal_img_path.endswith('.PNG'):
            gal_img = cv2.imread(gal_path + '/' + gal_folder + '/' + gal_img_path)
            gal_img,gal_std_img = cv2_to_torch(gal_img,do_extract = True)
            gal_img = gal_img.cuda()
            with torch.no_grad():
                gal_img_embd = resnet50(gal_img)[0,:,0,0].cpu().data.numpy() if querry_with_backbone else resnet50(gal_img)[0].cpu().data.numpy()
            gal_embds[gal_path + '/' + gal_folder + '/' + gal_img_path] = gal_img_embd
            gal_std_imgs[gal_path + '/' + gal_folder + '/' + gal_img_path] = gal_std_img

# np.save('/home/lam/CYBER/Factory/ShapeClassification/template_emb.npy', gal_embds)
idx = 0
num_test = 0
num_corr = 0
for query_path,query_emb in query_embds.items():
    print(idx)
    query_label = query_path.split('/')[-2].split('.')[0].split('_')[0]
    sim_dict = {}
    for gal_path, gal_embd in gal_embds.items():
        sim_dict[gal_path] = cosine_sim(gal_embd, query_emb)
    sim_dict = sorted(sim_dict.items(), key=lambda x: x[1])
    # print('Top 5 score : ' + str(np.array(sim_dict)[-5:,1][::-1]))
    if not os.path.exists('result/' + str(idx)) :
        os.mkdir('result/' + str(idx))
    top1_label = sim_dict[-1][0].split('/')[-2].split('.')[0].split('_')[0]
    top2_label = sim_dict[-2][0].split('/')[-2].split('.')[0].split('_')[0]
    top3_label = sim_dict[-3][0].split('/')[-2].split('.')[0].split('_')[0]  
    cv2.imwrite('result/' + str(idx) + '/' + 'query.png', query_std_imgs[query_path])
    cv2.imwrite('result/' + str(idx) + '/' + 'top1.png', merge_im(query_std_imgs[query_path],gal_std_imgs[sim_dict[-1][0]],query_label,top1_label))
    cv2.imwrite('result/' + str(idx) + '/' + 'top2.png', merge_im(query_std_imgs[query_path],gal_std_imgs[sim_dict[-2][0]],query_label,top2_label))
    cv2.imwrite('result/' + str(idx) + '/' + 'top3.png', merge_im(query_std_imgs[query_path],gal_std_imgs[sim_dict[-3][0]],query_label,top3_label))
    if query_label == top1_label : #or query_label == top2_label or query_label == top3_label :
        num_corr +=1
    else:
        print('True label : ' + str(query_label))
        print('Predict label : ' + str(top1_label))
        
    num_test +=1
    idx += 1
print(num_corr/num_test)
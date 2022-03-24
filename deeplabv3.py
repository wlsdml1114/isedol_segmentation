import os
import cv2
import torch
from tqdm import tqdm
from demo.image_matting.colab.inference2 import run
from PIL import Image
from torchvision import transforms
import numpy as np
file_path = '/home/jini1114/git/MODNet/input'
temp_dir = '/home/jini1114/git/MODNet/output'
path_in = '/home/jini1114/git/MODNet/temp'
path_out = '/home/jini1114/git/MODNet/mp4'
fps = 60

model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

def inference(input_image) :
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image  = np.array(input_image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.shape[:2])
    r.putpalette(colors)
    return r

input_lists = os.listdir(file_path)

for input_list in input_lists:
    if not(os.path.exists(temp_dir)) : 
        os.system('mkdir -p ' + temp_dir)
    print(input_list,'start')
    if not(os.path.exists(path_in)) : 
        os.system('mkdir -p ' + path_in)
    cap = cv2.VideoCapture(os.path.join(file_path,input_list))
    count = 0
    while True :
    
        if (count%1000 == 0):
            print(count)
        ret, frame = cap.read()

        if ret == False :
            break

        if input_list[0] == 'l' :
            cv2.imwrite(os.path.join(temp_dir,'%d.jpg'%(count)),frame[500:,:1000,:])
        if input_list[0] == 'r' :
            cv2.imwrite(os.path.join(temp_dir,'%d.jpg'%(count)),frame[500:,1000:,:])
        count+=1
    
    files = os.listdir(temp_dir)
    for idx in range(len(files)) :
        img = cv2.imread(os.path.join(temp_dir,'%d.jpg'%(idx)))
        res = inference(img)
        cv2.imwrite(os.path.join(path_in,'%d.jpg'%(idx)),res)
        

    frames = []

    files = os.listdir(path_in)
    for idx in tqdm(range(len(files)),desc = 'frame loading'):
        img = cv2.imread(os.path.join(path_in,'%d.png'%(idx)))
        frames.append(img)
    h,w,l = img.shape
    size = (w,h)

    output = cv2.VideoWriter(os.path.join(path_out,'matte_'+input_list),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

    for i in tqdm(range(len(frames)),desc = 'mp4 making'):
        output.write(frames[i])

    output.release()


    frames = []

    files = os.listdir(path_in)
    for idx in tqdm(range(len(files)),desc = 'frame loading'):
        img = cv2.imread(os.path.join(temp_dir,'%d.jpg'%(idx)))
        frames.append(img)

    output_cut = cv2.VideoWriter(os.path.join(path_out,'cut_'+input_list),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

    for i in tqdm(range(len(frames)),desc = 'mp4 making'):
        output_cut.write(frames[i])

    output_cut.release()


    os.system('rm -r '+temp_dir)
    os.system('rm -r '+path_in)

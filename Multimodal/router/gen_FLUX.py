from fastapi import APIRouter
from fastapi.responses import FileResponse
from urllib import request, parse
import json
import random
import glob
from PIL import Image
import asyncio


### loading workflow ###
server_address = "127.0.0.1:9919"
f = open("./router/ComfyUi_wf/flux_api.json")
prompt = json.load(f)

############################# Server ########################
#################### FLUX 이미지 생성 #########################

router = APIRouter(prefix="/FLUX")

@router.get("/")
def _():
    return {"PAGE":"This Server is FLUX -> generate image"}

### start method ###
def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://"+server_address + "/prompt", data=data)
    request.urlopen(req)
    
@router.post("/FLUX")
async def generate_image(text:str = "A cute cat is holding a smartphone and playing a game. The cat is sitting on a single sofa and wearing a knitwear."):
    poo = 0
    
    # input_text
    prompt['21']['inputs']['text'] = text + "The background behind the image is white"

    # KSampler
    prompt['19']["inputs"]["seed"] = random.randint(1, 1000000000)
    prompt['19']["inputs"]["steps"] = 8

    # Empty Latent Image
    prompt["34"]["inputs"]['width'] = 1024
    prompt["34"]["inputs"]['height'] = 1024

    # set image name 
    prompt["24"]["inputs"]["filename_prefix"] = "sample_image"
    
    queue_prompt(prompt) # start queue
    
    await asyncio.sleep(40)  # 필요한 시간(초)을 대기
    
    path = fr"C:\Users\Admin\Desktop\comfyUI\ComfyUI\output\sample_image*.png"
    pt = glob.glob(path)
    
    return FileResponse(pt[-1], media_type='image/png', filename='image.png') 


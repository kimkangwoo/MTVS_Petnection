from fastapi import APIRouter, File, UploadFile,HTTPException
from PIL import Image
from urllib import request, parse
import json
import glob
import os
import io
from fastapi.responses import FileResponse

print(os.getcwd())

router = APIRouter(prefix='/3d_modeling')

LOG = "[LOG] comfy_req.py |"
ADDRESS = "127.0.0.1:9919"
PATH = 'router/ComfyUi_wf/stable_3D.json'

# Start prompt
def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://"+ADDRESS + "/prompt", data=data)
    request.urlopen(req)

# Loading workflow
with open(PATH, 'r', encoding='utf-8') as f:
    workflow = json.load(f)

# image to 3d generation
@router.post("/")
async def main(image: UploadFile = File(...), image_name = "test_image.png"):
    
    try:
        image_bytes = await image.read()# UploadFile 데이터를 바이트로 읽기
        image_stream = io.BytesIO(image_bytes)# 바이트 데이터를 BytesIO로 변환
        img = Image.open(image_stream)# BytesIO 객체로 이미지 열기
        
    except Exception as e:
        print(f"Error: {e}")

    # input dir save img
    path = f"ComfyUI/input/{image_name}"
    img.save(path)
    print(LOG, "정상적으로 저장되었습니다.")
    workflow['16']['inputs']['image'] = image_name
    
    # image 2 3d
    queue_prompt(workflow)
    
    # select the latest .glb
    glb_file_path = f"ComfyUI/output/*.glb"
    glb_file_path = glob.glob(glb_file_path)
    return FileResponse(glb_file_path[-1], media_type='model/gltf-binary', filename='file.glb')
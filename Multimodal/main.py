# 서버
# URL : http://221.163.19.142:55508
# uvicorn main:app --reload --port 55508 --host 0.0.0.0 

# comfyui 실행
# cd ComfyUI
# python main.py --port 9919 --cuda-device 0
############################################################
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def main():
    return  {"PAGE":"this page main page"}

# Vision Multimodal
from router import molmo_chat
app.include_router(molmo_chat.router)

# comfyUI SD1.5 - sketch router
from router import sketch_SD15
app.include_router(sketch_SD15.router)

# comfyUI FLUX router
# from router import gen_FLUX
# app.include_router(gen_FLUX.router)

# comfyUI 3D router
# from router import comfy_req
# app.include_router(comfy_req.router)

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run("main:app", host="0.0.0.0", port=55508)

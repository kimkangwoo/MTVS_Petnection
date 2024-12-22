from fastapi import FastAPI
from app.router.texture_maker import texture_router
from app.router.obj_maker import obj_router
from app.router.llm_tester import llm_router

from fastapi.staticfiles import StaticFiles

app = FastAPI()
@app.get('/')
def main():
    return {"message":'텍스처 생성 서버'}

UPLOAD_DIR = 'app/database/input_images'

app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

texture_router = app.include_router(texture_router)
obj_router = app.include_router(obj_router)
llm_router = app.include_router(llm_router)

# ngrok http 8080 --domain sheepdog-bold-bulldog.ngrok-free.app
# https://sheepdog-bold-bulldog.ngrok-free.app
# equal-seasnail-stirred.ngrok-free.app 
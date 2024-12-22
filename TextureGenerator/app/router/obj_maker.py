from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import os

obj_router = APIRouter(prefix='/3dobj')

### 3d 오브젝트 생성 및 송신을 담당하는 코드입니다.

## 테스트용 fbx 파일 보내기

@obj_router.get('/send-fbx')
async def send_fbx():
    '''
    fbx 파일 송신 테스트용 api 입니다.
    '''
    fbx_file_path = 'app/database/fbx_files.fbx'
    if not os.path.exists(fbx_file_path):
        raise HTTPException(status_code=404, detail="FBX file not found")
    return FileResponse(fbx_file_path, media_type='application/octet-stream', filename="test.fbx")
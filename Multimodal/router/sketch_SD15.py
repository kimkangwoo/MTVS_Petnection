from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import websocket
import uuid
from urllib import request, parse
import json 
import random
from PIL import Image
import io
import time
import httpx



LOG = "| sketch_SD15.py | LOG |"
############################# Method ################################

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(request.urlopen(req).read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break #Execution is done
                    else:
                        current_node = data['node']
        else:
            if current_node == '453':
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output

    return output_images

############################# Server ################################
#################### SD1.5 sketch 이미지 생성 #########################

router = APIRouter(prefix="/SD15")

# JSON 파일 불러오기
file_name = "./router/ComfyUi_wf/sketch_lora.json"

with open(file_name, 'r', encoding='utf-8') as file:
    data = json.load(file)
    
# init
server_address = "127.0.0.1:9919"
client_id = str(uuid.uuid4())


@router.post("/gen_img_test", tags=["Stable Diffusion"])
def gen_img_test(neg_tex:str = "ugly",
            pos_tex:str = "", 
            seed_ran:bool = True):
    if seed_ran : 
        data['93']['inputs']['seed'] = random.randint(0, 1e15) # seed 랜덤
    else :
        data['93']['inputs']['seed'] = 674482827147390
    data['94']['inputs']['text'] =  neg_tex # 부정 텍스트 입력
    # data['96']['inputs']['text'] = "(masterpiece:1.2), (best quality:1.3), (sketch:1.6), (ink:1.45), (paper:1.45), no humans, (tail:1.3), (cute:1.2), (center:1.5), (same eyes:1.3)" + pos_tex # 긍정 텍스트 입력
    data['96']['inputs']['text'] = pos_tex # 긍정 텍스트 입력
    data["443"]["inputs"]['height'] = 720
    data["443"]["inputs"]["width"] = 480
    
    try:
        ws = websocket.WebSocket()
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
        images = get_images(ws, data)  # 단일 이미지 객체 반환
        ws.close()
    except Exception as e:
        return {"error": f"WebSocket connection failed: {str(e)}"}

    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
    
    # 이미지 파일로 저장 후 반환
    image_path = "generated_image.png"
    image.save(image_path, format="PNG")  # 단일 이미지 객체이므로 바로 저장 가능
    
    return FileResponse(image_path, media_type="image/png", filename="generated_image.png")


molmo_url = 'http://221.163.19.142:55508/chatbot/chat_testing_prompt'
genimg_url = 'http://221.163.19.142:55508/SD15/gen_img_test'

text = "Animals in the current video Describe the animal's characteristics, such as fur color, fur pattern, and fatness"
pos_prompt = ' circle, round, centered, in the center, (masterpiece, best quality:1.1), (sketch:1.5), (paper:1.3), no humans, whiskers, tail, big eyes, cute\n\n'
neg_prompt = '(embedding:easynegative:1.2), (embedding: badhandv4:1.2), 1girl, solo, lowres, artist name, signature, watermark, low contrast'
tail = "bad anatomy, extra tail, multiple tails, poorly drawn tail, distorted tail, misplaced tail, extra limbs, bad proportions, disfigured"

@router.post("/gen_image", tags=["Stable Diffusion"])
async def gen_image2(image: UploadFile):
    """
    현재 엔드포인트는 이미지를 넣으면 해당하는 이미지를 분석하여 스케치한 이미지를 만듭니다.
    
    소요시간 : 15 ~ 40초 
        -> 이미지 분석 : 10초
        -> 이미지 생성 : 5초 
    """
    st = time.time()
    # Byte code -> image
    if image.content_type.startswith("image/"):
        print(LOG, "정상적으로 이미지가 들어왔습니다.", "|") 
        
        # 이미지 열기
        image_bytes = await image.read()
        image_io = io.BytesIO(image_bytes)
        image = Image.open(image_io)
        
        # RGBA 모드일 경우 RGB로 변환
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        
        # 바이트 배열로 이미지 저장
        ioByte = io.BytesIO()
        image.save(ioByte, format='JPEG')
        ioByte.seek(0)
        
        
        # 이미지 추론하기
        print(LOG, "이미지를 추론합니다.", "|") 
        params = {'text': text}
        files = {"image" : ("testimage.jpg", ioByte, 'image/jpeg')}
        
        timeout = httpx.Timeout(60.0)  # 60초 타임아웃 설정
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(molmo_url, params=params, files=files)
                response.raise_for_status()  # 상태 코드가 4xx나 5xx이면 예외 발생
            except httpx.ReadTimeout:
                print("서버 응답이 너무 늦어 타임아웃이 발생했습니다.")
            except httpx.RequestError as e:
                print(f"요청 중 오류가 발생했습니다: {e}")
            except Exception as e:
                print(f"알 수 없는 오류가 발생했습니다: {e}")
        print(LOG, "이미지를 추론완료.", "|") 
        
        # 추론값으로 이미지 생성하기
        params = {
            'neg_tex': neg_prompt + tail,
            'pos_tex': pos_prompt + response.json(),
            'seed_ran': 'false'   
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(genimg_url, params=params)
                response.raise_for_status()  # 상태 코드가 4xx나 5xx이면 예외 발생
                image =  response.content
                end = time.time()
                print(LOG, "소요 시간 :", int(end)-int(st), "초 |") 
                
                with open("image.png", "wb") as f:
                    f.write(image)
                return FileResponse("image.png", media_type="image/png")
            
            except httpx.ReadTimeout:
                return {"error":"서버 응답이 너무 늦어 타임아웃이 발생했습니다."}
            except httpx.RequestError as e:
                return {"error" : f"요청 중 오류가 발생했습니다: {e}"}
            except Exception as e:
                return {"error": f"알 수 없는 오류가 발생했습니다: {e}"}          
        

    else : 
        return {"ERROR":"I can't search image"}
    

import json
from urllib import request
import time
import os
from PIL import Image
import io
import urllib
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


workflow_path = "./workflows/workflow_default.json"

# 파일에서 프롬프트 로드

#text = "Create a close-up image of animal fur with a solid blure color. The fur is glossy, sleek, without any patterns or additional colors. The texture should reflect light slightly to emphasize the smooth and shiny appearance of the black fur. There should be no visible background or additional elements, focusing solely on the animal's sleek black fur."
negative_text = "no animal shape,no face,no paws,no eyes,no nose,no ears,no limbs,no body,no full animal"

BASE_URL = os.getenv('API_BASE_URL', 'http://127.0.0.1:8188')#내꺼 http://127.0.0.1:8188/// 서울 서버 http://221.163.19.142:55509
IMAGE_SAVE_PATH = os.getenv('IMAGE_SAVE_PATH', 'app/database/generation_img/')


# API로 요청 전송
def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"{BASE_URL}/prompt", data=data)
    request.urlopen(req)

    # API 응답 받기
    with request.urlopen(req) as response:
        result = response.read().decode('utf-8')
        result_data = json.loads(result)
        print("이미지 생성 요청이 전송되었습니다.")
        return result_data

#작업 상태 확인
def check_prompt_status(prompt_id):
    # 상태 확인 URL (정확한 경로 확인 필요)
    url = f"{BASE_URL}/history/{prompt_id}"
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            result = response.read().decode('utf-8')
            result_data = json.loads(result)
            return result_data
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}, {e.reason}")
        return None
    

def download_image(filename):
    # 이미지 파일의 URL 구성
    url = f"{BASE_URL}/view?filename={filename}&type=output"
    try:
        # 이미지 데이터를 다운로드
        response = urllib.request.urlopen(url)
        image_data = response.read()

        # 메모리 내에서 이미지 읽기
        image = Image.open(BytesIO(image_data))

        # 파일로 저장
        # filename 대신에 사용할 경로 지정하기
        filepath = os.path.join(IMAGE_SAVE_PATH, filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        logging.info(f"이미지 다운로드 완료: {filename}")
        return image

    except Exception as e:
        logging.error(f"이미지 다운로드 중 오류 발생: {str(e)}")

def load_and_update_workflow(filepath, updates):
    with open(filepath, "r") as f:
        workflow_data = json.load(f)
    for key, value in updates.items():
        workflow_data[key]['inputs']["text"] = value
    return workflow_data



def defualt_img_gen(wokrflow_path,text):
    # workflow 파일 열기
    with open(wokrflow_path, "r") as f:
        workflow_data = json.load(f)
    workflow_data["2"]['inputs']["text"] = text
    workflow_data["3"]['inputs']["text"] = negative_text
    response_data = queue_prompt(workflow_data)
    
    prompt_id = response_data['prompt_id']
    
    
    
    while True:
        ### 실행 완료 확인하기
        status = check_prompt_status(prompt_id)

        ### 완료 되었으면 다운 받기
        if status and prompt_id in status and status[prompt_id]['status']['status_str'] == 'success':
            print("이미지 생성 완료")
            filename = status[prompt_id]['outputs']['7']['images'][0]['filename']
            image = download_image(filename)
            break
        else:
            print("이미지 생성 중... 다시 확인합니다.")
            time.sleep(2)
    
    
    return image

def wait_download(prompt_id, download_key='109', output_key='images'):
    while True:
        status = check_prompt_status(prompt_id)
        if status and prompt_id in status and status[prompt_id]['status']['status_str'] == 'success':
            print("이미지 생성 완료")
            filename = status[prompt_id]['outputs'][download_key][output_key][0]['filename']
            return download_image(filename)
        else:
            print("이미지 생성 중... 다시 확인합니다.")
            time.sleep(2)





'''
imgae_data 확인용 코드였음.
image_data = defualt_img_gen(workflow_path,text)

image = Image.open(io.BytesIO(image_data))
image.show()

if image_data:
    print(f"이미지 데이터 크기: {len(image_data)} 바이트")
    if len(image_data) > 0:
        print("이미지가 성공적으로 다운로드되었습니다.")
    else:
        print("이미지가 비어 있습니다.")
else:
    print("이미지 데이터가 없습니다.")
'''
def inpainting2part(main):
    """
    특별한 무늬가 없는 고양이 텍스처를 생성하는 함수
    """

    workflow_data = load_and_update_workflow("app/database/workflows/nomal_cat_v6.json", {
    "2": main,
    "44":main
    })

    response_data = queue_prompt(workflow_data)
    
    prompt_id = response_data['prompt_id']
    image = wait_download(prompt_id, download_key='40')

    return image
    

def tabby_cat(main,stripe):
    """
    호랑이 줄무늬를 가진 고양이 텍스처를 생성하는 함수
    """

    workflow_data = load_and_update_workflow("app/database/workflows/tabby_cat_v13.json", {
    "2": main,
    "27": stripe,
    "51":main
    })

    response_data = queue_prompt(workflow_data)

    prompt_id = response_data['prompt_id']
    image = wait_download(prompt_id, download_key='46')

    return image

def calico_cat(main,stripe):
    """
    점박이 고양이 텍스처를 생성하는 함수
    """
    workflow_data = load_and_update_workflow("app/database/workflows/calico_cat_v2.json", {
    "2": main,
    "27": stripe,
    "51":main
    })

    response_data = queue_prompt(workflow_data)

    prompt_id = response_data['prompt_id']
    image = wait_download(prompt_id, download_key='46')

    return image

def vita_cat(main,stripe):
    """
    비타를 위한 특수 텍스처
    """
    workflow_data = load_and_update_workflow("app/database/workflows/vita_v1.json", {
    "2": main,
    "27": stripe,
    "51":main
    })

    response_data = queue_prompt(workflow_data)

    prompt_id = response_data['prompt_id']
    image = wait_download(prompt_id, download_key='94')

    return image
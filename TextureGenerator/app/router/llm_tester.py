from fastapi import APIRouter, Form, File, UploadFile
from app.utills.llm_utill import get_llm_response,load_llm_model,format_prompt,prepare_prompt_template,get_promptLLM_response,get_descriptionLLM_response
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
import json

from dotenv import load_dotenv
from PIL import Image
import io
import os
load_dotenv()
import base64
from urllib.parse import quote


UPLOAD_DIR = 'app/database/input_images'

llm_router = APIRouter(prefix='/llm_test')

memory_store = {}
'''
if id not in memory_store:
    memory_store[id] = ConversationBufferMemory()

memory = memory_store[id]'''

#text = " "
# 1단계: LLM 모델 로드
model = load_llm_model()


# 2단계: 시스템 프롬프트 준비 (필요에 따라 이 프롬프트를 수정하세요)
system_prompt = """
당신은 Text내용을 참고하여 이미지를 생성하는 프롬프트를 작성하는 어시스턴트입니다.
Text내용에서 동물의 부위별 털 특징을 이용하여 부위별로 프롬프트를 작성하고 json 형태로 답변하세요.
아래에 내용과 Text,Json을 기반으로 답변하세요.
1. Json에 내용을 참고하여 답변을 json 형태로 출력하세요.
2. json의 키 순서는 main, head, back, chest, tail, leg 순서로 작성하세요.
3. 키 값에는 ' ' 같은 공백을 포함시키지마세요.
4. Text 내용에서 부위별 털 특징을 이용하여 부위별로 이미지를 생성하는 프롬프트를 작성하세요.
5. 이미지를 생성할 때 털의 질감이 뚜렷하게 나오게 작성하는 문구를 추가하세요.
6. json 형식은 "animal_features": 로 시작하세요.

Text:{text}
Json:{context}
Answer:
"""

json_data = """
{
  "animal_features": {
    "head": {
      "generation_prompt": "Generate an image filled with smooth brown and white fur in a striped pattern, with brown as the primary color and hints of white interwoven. The fur should have a realistic texture, giving a soft and natural appearance."
    },
    "back": {
      "generation_prompt": "Generate an image filled with dense, coarse gray fur, solid in color and texture, capturing the rugged look and natural thickness of the fur."
    },
    "chest": {
      "generation_prompt": "Generate an image filled with soft, fluffy white fur, creating a light and airy texture that covers the entire image, giving a sense of gentle and full-bodied fur."
    },
    "legs": {
      "generation_prompt": "Generate an image filled with smooth brown and white fur with a spotted pattern, blending both colors seamlessly to create a realistic, natural fur texture."
    },
    "tail": {
      "generation_prompt": "Generate an image filled with thick, fluffy black and gray fur in a gradient effect, where the fur naturally transitions from black to gray, giving a rich and layered appearance."
    }
  }
}
"""


'''
# 3단계: 프롬프트 템플릿 준비
custom_prompt = prepare_prompt_template(system_prompt)

# 4단계: (선택 사항) 대화 기록과 컨텍스트 로드
conversation_history = memory.load_memory_variables({})
#context = openai_retriever.invoke(text)  # 또는 다른 컨텍스트를 사용하세요

# 5단계: 필요한 변수로 프롬프트를 포맷팅
formatted_prompt = format_prompt(
    custom_prompt,
    history=conversation_history,
    context=text,
    text="동물과 사람에 대한 토론주제를 제공해주세요."
)
'''
'''
# 6단계: 체인을 실행하고 응답 받기
response = get_llm_response(model,formatted_prompt)

# 7단계: 필요한 대로 응답 처리
print(response)
'''

def process_image(image_file: UploadFile):
    image = Image.open(image_file.file)
    # 이미지를 LLM에 입력할 형식으로 변환하거나, 이미지의 정보를 추출 (이미지 인식 모델 활용 가능)
    # 예를 들어 이미지를 텍스트 설명으로 변환하는 추가 모델이 있다면 이곳에서 호출 가능
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    return image_bytes.getvalue()


@llm_router.post('/llm_test')
def llm_test(id:int = Form(...),text:str=Form(...)):
    if id not in memory_store:
        memory_store[id] = ConversationBufferMemory()

    memory = memory_store[id]
    # 3단계: 프롬프트 템플릿 준비
    custom_prompt = prepare_prompt_template(system_prompt)

    # 4단계: (선택 사항) 대화 기록과 컨텍스트 로드
    conversation_history = memory.load_memory_variables({})
    #context = openai_retriever.invoke(text)  # 또는 다른 컨텍스트를 사용하세요
    # 5단계: 필요한 변수로 프롬프트를 포맷팅
    formatted_prompt = format_prompt(
        custom_prompt,
        history=conversation_history,
        context=json_data,
        text=text
    )

    # 6단계: 체인을 실행하고 응답 받기
    response = get_llm_response(model,custom_prompt,formatted_prompt)

    # 7단계: 필요한 대로 응답 처리
    print(response)
    
    clean_response = response.replace("json\n", "").strip()
    clean_response = clean_response.replace("```", "").strip()
    resopnse_json = json.loads(clean_response)
    resopnse_json
    return {'topic':resopnse_json["animal_features"]["main"]}


img_system_prompt = """
당신은 이미지를 확인하고 어떤 동물인지 설명하는 어시스턴트입니다.
아래에 내용을 참고하여 답변하세요.

Question: {text}
Context: {context}
Image URL: {image_url}

Answer:
"""
def encode_image(image_file: UploadFile):
    """이미지를 base64로 인코딩하는 함수"""
    image_bytes = image_file.file.read()
    return base64.b64encode(image_bytes).decode('utf-8')


'''

@llm_router.post('/llm_img_test')
async def llm_img_test(
    id: int = Form(...), 
    text: str = Form(...), 
    image: UploadFile = File(None)
):
    """
    토큰 수 제한 때문에 이미지를 URL로 전달하기 위해 서버에 저장하고,
    그 경로를 반환하는 방법을 사용합니다.
    """
    if id not in memory_store:
        memory_store[id] = ConversationBufferMemory()

    memory = memory_store[id]
    # 3단계: 프롬프트 템플릿 준비
    custom_prompt = prepare_prompt_template(img_system_prompt)

    # 4단계: 대화 기록과 컨텍스트 로드
    conversation_history = memory.load_memory_variables({})
    
    # 5단계: 이미지가 있는 경우 이미지를 처리
    image_description = ""
    image_url = ""
    if image:
        # 이미지 파일을 저장할 경로
        file_location = os.path.join(UPLOAD_DIR, image.filename)

        # 이미지 URL 생성 (여기서는 로컬 서버 URL로 설정, 실제 서버에 맞게 변경해야 함)
        image_url = f"https://sheepdog-bold-bulldog.ngrok-free.app/images/{image.filename}"
        print(image_url)
        image_description = "이미지가 포함되었습니다."

        
    # 프롬프트 포맷팅
    formatted_prompt = format_prompt(
        custom_prompt,
        history=conversation_history,
        context=' ',
        image_url = image_url,
        text=text,
    )

    # 7단계: 체인을 실행하고 응답 받기
    response = get_llm_response(model, custom_prompt, formatted_prompt, image_url=image_url)

    # 8단계: 응답 처리 및 반환
    print(response)
    return {'topic': response}
'''

@llm_router.post('/llm_img_test')
async def llm_img_test(
    id: int = Form(...), 
    text: str = Form(...), 
    image: UploadFile = File(None)
):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    file_location = os.path.join(UPLOAD_DIR, image.filename)

    with open(file_location, "wb") as f:
        f.write(await image.read())

    encoded_filename = quote(image.filename)

    image_url = f"https://sheepdog-bold-bulldog.ngrok-free.app/images/{encoded_filename}"

    

    from openai import OpenAI

    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s in this image?"},
            {
            "type": "image_url",
            "image_url": {
                "url": image_url,
            },
            },
        ],
        }
    ],
    max_tokens=300,
    )

    print(response.choices[0])
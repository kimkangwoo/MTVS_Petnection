# 모듈 불러오기
from fastapi import APIRouter, File, UploadFile,HTTPException
from fastapi.responses import JSONResponse

from PIL import Image
import io
import json
import re

from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    GenerationConfig, 
    BitsAndBytesConfig
    )
from PIL import Image
import torch
from deep_translator import GoogleTranslator

# Langchain
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
load_dotenv()

import os

####################################################

LOG = "| molmo_chat.py | LOG |"
CUDA = 'cuda:0'

####################################################

# 자료구조 정의 (pydantic)
class FurDescribe(BaseModel):
    breed : str = Field(description="describe animal species")
    index : int = Field(description="""
                        matching the animal for predominantly color
                        1 == The main color is black,
                        2 == The main color is white,
                        3 == The main color is brown,
                        4 == It is mainly black, but other colors are available.
                        """)
    color : str = Field(description="what is main color")
    
# load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True, # 원격에서 커스텀 코드를 가져와 실행
    torch_dtype='auto',
    # torch_dtype=torch.float16,
    device_map={"": CUDA}
)

# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,   # 4비트로 로드
    bnb_4bit_use_double_quant=True,  # 더블 양자화 사용
    bnb_4bit_quant_type="nf4",  # 양자화 유형 선택 (nf4 또는 fp4)
    bnb_4bit_compute_dtype=torch.float16  # 계산을 위한 dtype (float16을 자주 사용)
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True, # 원격에서 커스텀 코드를 가져와 실행
    torch_dtype='auto',
    quantization_config=bnb_config,  # 양자화 설정 전달
    device_map={"": CUDA}, 
)

# 모델 불러오기
open_model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
device = torch.device(CUDA if torch.cuda.is_available() else "cpu")

########################### chatbot 서버 구문 ####################################

router = APIRouter(prefix = "/chatbot")

@router.get('/', tags=['/Multimodal'])
def main():
    return {"PAGE":"this page vision chatbot page"}

@router.post('/chat', tags=['/Multimodal'],
             description="이미지를 받고 이미지에 대한 설명을 chatbot이 해줍니다. \nchatbot NPC가 와서 발랄하게 현재 이미지의 동물에 대해서 설명해줄거에요.")
async def main(user:int, image: UploadFile = File(...)):
    userName = user
    if image.content_type.startswith("image/"):
        # 파일을 바이너리 데이터로 읽음
        print(LOG, '정상적으로 이미지가 들어왔습니다.', "|")
        image_bytes = await image.read()    
        image = Image.open(io.BytesIO(image_bytes))

        # RGBA 모드일 경우 RGB로 변환
        if image.mode == 'RGBA':
            image = image.convert("RGB")

        # Text = "Look at the pose and expression of this animal! How do you think it's feeling? Happy, curious, shy? In 150 characters or so, say it in a friendly and playful way. Note here: no emoticons and absolutely no special characters."
        Text = "describe the image focus on animal pose and emotions"
        inputs = processor.process(images=image, text=Text)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        print(LOG, '추론 시작', "|")
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
        
        print(LOG, '추론 종료', "|")
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        text = text.replace('\n', '')
        
        # 프롬프트 생성
        system_template = "현재 분석한 사진의 동물이 어떤 심정인지 한글로 100자 내외로 재밌고 친근하게 얘기해주세요.이모티콘은 필요없어요 현재 상황에만 대해서 말해주세요 마지막 줄에 <br><br><color=#1A248D><b>#내새끼</b></color> 형식으로 해시태그를 3개 주세요"
        prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{text}")])
        
        # 체인 생성
        chain = prompt_template | open_model | parser
        
        print(LOG, 'OpenAI 추론', "|")
        
        # 추론하기 및 String 형식으로 return
        predict = chain.invoke(text)
        
        predict = predict.replace('\n', '')
        predict = predict.replace('\"', '')
        
        print(repr(predict))
        
        return predict
    else : 
        return {"ERROR": "wrong input"}
    
@router.post('/im_text', tags=['/Multimodal'])
async def main(image: UploadFile = File(...)):
    print(image.content_type)

    text = """
Analyze the image and extract the following information about the animal's fur:

1. What is the overall color of the animal's fur?
    - Describe the dominant color(s) of the fur.
2. Describe the texture and length of the animal's fur (e.g., smooth, coarse, short, long).
    - Is the fur smooth or coarse? Is it short or long?
3. Does the animal have multiple colors distinctly separated into patches?
    - If yes, describe the colors and their approximate locations on the body (e.g., white chest, orange back, black spots).
    - Does the color pattern resemble a Calico pattern (distinct patches of white, orange, and black)?
4. Does the animal have any visible stripe or pattern on its fur? If yes, describe the pattern type and indicate the color of the stripes or pattern.
    - For example: Are the stripes thin and continuous, resembling a mackerel pattern, or are they broader?
    - If applicable, describe the Tabby pattern (e.g., Mackerel, Classic, or Spotted Tabby).
5. Is the fur of the same color throughout the entire body?
    - If yes, describe the single color and confirm whether it is a Solid Color.
6. Identify the fur color for specific body parts:
    - Legs:
    - Belly:
    - Chest:
    - Back:
    - Face:
    """

    if image is None:
        raise HTTPException(status_code=400, detail="No image file uploaded")
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    
    if image.content_type.startswith("image/"):
        # 파일을 바이너리 데이터로 읽음
        print(LOG, '정상적으로 이미지가 들어왔습니다.', "|")
        image_bytes = await image.read()    
        image = Image.open(io.BytesIO(image_bytes))
        
        # RGBA 모드일 경우 RGB로 변환
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        
        
        Text = text
        #print(Text)
        inputs = processor.process(images=image, text=Text)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        print(LOG, '추론 시작', "|")
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
        
        print(LOG, '추론 종료', "|")
        generated_tokens = output[0,inputs['input_ids'].size(1):]

        return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else : 
        return {"ERROR": "wrong input"}
    
@router.post('/chat_testing_prompt', tags=['/Multimodal'])
async def main(text:str, image: UploadFile = File(...)):
    if image.content_type.startswith("image/"):
        # 파일을 바이너리 데이터로 읽음
        print(LOG, '정상적으로 이미지가 들어왔습니다.', "|")
        image_bytes = await image.read()    
        image = Image.open(io.BytesIO(image_bytes))
        
        # RGBA 모드일 경우 RGB로 변환
        if image.mode == 'RGBA':
            image = image.convert("RGB")

        inputs = processor.process(images=image, text=text)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        print(LOG, '추론 시작', "|")
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
        
        print(LOG, '추론 종료', "|")
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        
        return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else : 
        return {"ERROR": "wrong input"}
    
@router.post('/index_of_fur_color', tags=['/Multimodal'])
async def main(image: UploadFile = File(...)):
    """### 엔드포인트 설명 ###

    Args >>>
        image : 동물색을 판정할 수 있게 동물이 확실하게 보이는 사진 1장
    
    JSON_structure >>> <br>
        {
            
            breed : 동물 종류를 말해줍니다. (일관적이지 않은 영어)<br>
            index : 동물의 인덱스를 말해줍니다. (1.검정 / 2.흰색 / 3.갈색 / 4. 블랙탄)<br>
            color : 눈이 조금 흐릿한 AI가 판단한 색깔을 말해줍니다. (일관적이지 않은 영어)
               
        }
    """
    if image.content_type.startswith("image/"):
        # 파일을 바이너리 데이터로 읽음
        print(LOG, '정상적으로 이미지가 들어왔습니다.', "|")
        image_bytes = await image.read()    
        image = Image.open(io.BytesIO(image_bytes))
        
        # RGBA 모드일 경우 RGB로 변환
        if image.mode == 'RGBA':
            image = image.convert("RGB")

        inputs = processor.process(
            images=image,
            text="Explain the species and fur color of the animal based on the primary color of each part.")
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        print(LOG, '추론 시작', "|", "동물 털 패턴 분석")
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
        
        print(LOG, '추론 종료', "|", "동물 털 패턴 분석")
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        
        text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
        # 출력 파서 정의
        output_parser = JsonOutputParser(pydantic_object=FurDescribe)
        format_instructions = output_parser.get_format_instructions()

        # prompt 구성
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": format_instructions},
        )
        
        print(LOG, 'GPT-4O-mini 추론 시작', "|", "동물 털 패턴 분석")
        chain = prompt | open_model | output_parser
        print(LOG, 'GPT-4O-mini 추론 종료', "|", "동물 털 패턴 분석")
        
        return JSONResponse(content=chain.invoke({"query": text}))
    else : 
        return {"ERROR": "wrong input"}
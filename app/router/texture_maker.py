from fastapi import APIRouter, File, UploadFile, Form, HTTPException,Depends
from fastapi.responses import FileResponse
import os
import shutil
from PIL import Image
from io import BytesIO
from app.utills.texture_utill import extract_mesh, check_duplicate_uv, generate_texture
from app.utills.imggen_utill import defualt_img_gen, queue_prompt,inpainting2part,tabby_cat,calico_cat,vita_cat
from app.utills.llm_utill import load_llm_model,format_prompt,prepare_prompt_template,get_llm_response,get_promptLLM_response,get_descriptionLLM_response

import json
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
import requests


texture_router = APIRouter(prefix='/texture')

# 이 곳은 텍스처를 생성하는 작업을 하는 서버입니다. 


db_dir = 'app/database/'
glb_dir = db_dir+'glb_files/'
img_dir = db_dir+'input_images/'
ful_dir = db_dir+'tmp_ful_img/'
texture_dir = db_dir+'output_textures/'

tmp_ful = ful_dir + '고양이털.png'

glb_dic = {}

key_count = 0

# 로컬 파일에 모델링 번호로 딕셔너리 생성
for filename in sorted(os.listdir(glb_dir)):
    glb_dic[key_count] = glb_dir + filename
    key_count +=1

'''
class PetMeshRequest(BaseModel):
    image: UploadFile
    petMeshNumber:int
    userId:int
pet_request: PetMeshRequest = Depends()
'''


json_data = """
{
  "animal_type": "Cat",
  "fur_type": "Teddy",
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
    },
    "fur_pattern":{
      "generation_prompt": "A solid black background, completely dark and featureless. No textures, patterns, or lighting effects; the background should be uniformly black with a smooth, matte finish, creating a simple and unobtrusive setting."
    }
  }
}
"""



@texture_router.post('/img2texture222',tags=['texture'])
async def img2texture222(userId:int = Form(...),petMeshNumber:int = Form(...),image:UploadFile = File(...)):

    #print(pet_request.image.content_type)
    """
    이미지를 입력 받아 반려동물 특징을 추출하고 
    특징에 맞는 동물 털 이미지 생성합니다.
    모델링 넘버를 이용하여 glb 파일을 검색합니다.
    glb 파일에 메쉬 데이터를 추출하여 생성 이미지를 매핑하여 텍스처를 생성합니다.
    """
    try:
        """
        이미지를 입력받고 저장하는 코드
        """
        
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        file_location =os.path.join(img_dir,image.filename)

        with open(file_location,'wb') as b:
            shutil.copyfileobj(image.file,b)


        

        """
        이미지에서 특징을 추출하는 코드
        gpt 모델 사용해서 특징 추출하기
        """
        
        # 강우의 molmo의 연결해서 답변을 받기 반환 값은 feature_respone

        with open(file_location, "rb") as image_file:
            files = {'image': (image.filename, image_file, 'image/jpeg')}
            data = {"text": "Describe this image"}
            
            # 외부 API 호출
            response = requests.post("https://equal-seasnail-stirred.ngrok-free.app/chatbot/im_text", files=files)

        
        print("이미지 처리 완료")

        # 결과 출력
        print("추출된 텍스트:",response.text)


        # response를 llm에 보내서 프롬프트 생성하기
        model = load_llm_model()
        prompt = """
            You are an assistant that writes prompts for generating images.
            Refer to the characteristics of the animal's fur in the text and follow the guidelines below to write the prompt.

            1. Describe the detailed characteristics of the animal's fur. 
            For example, specify the fur's length, thickness, sheen, and whether it is curly or straight.

            2. The image should be filled entirely with the animal's primary fur color. Do not include any background or other elements, only the color of the fur.

            3. When describing colors, follow these guidelines:

            3-1. **Brightness**: Clearly indicate the brightness, such as 'light orange' or 'dark brown.'

            3-2. **Color combinations**: Specify color combinations clearly, such as 'orange and cream.'

            3-3. **Natural expression**: Use smooth and natural language when describing colors. For example, 'natural colors of light orange and cream' to describe the tone and combination of colors.

            4. Avoid using abstract or emotional expressions. 

            5. The image should be perfectly symmetrical from left to right.
            Instead of words like 'energy' or 'happy,' focus on specific visual characteristics.
        """


        
        custom_prompt = prepare_prompt_template(prompt)
       
        formatted_prompt = format_prompt(
            custom_prompt,
            text = response.text
        )

        print(formatted_prompt)

        feature_response = get_llm_response(model,custom_prompt,formatted_prompt)
        
        print("프롬프트 추출"+feature_response)
        """
        특징 기반 metrial 이미지 생성
        """
        
        #테스트용 특징
        #feature_respone = "Create a close-up image of animal fur with a solid blure color. The fur is glossy, sleek, without any patterns or additional colors. The texture should reflect light slightly to emphasize the smooth and shiny appearance of the black fur. There should be no visible background or additional elements, focusing solely on the animal's sleek black fur."
        #feature_respone = "Create a close-up image of animal fur with a light orange main color. The fur should have no visible patterns and be a solid color across the body. The fur should be short, smooth, and have bright but slightly muted colors for a more natural appearance"


        #임시용 feature_response
        #feature_response = "Create an image filled entirely with the natural colors of the animal's fur. The fur should be medium length, moderately thick, and possess a subtle sheen that catches the light. The primary color should be a natural blend of deep, dark brown with subtle hints of rich, golden undertones. This combination should be smooth and seamlessly integrated, creating a harmonious color palette. The image should maintain perfect symmetry from left to right, focusing solely on these specified fur colors without any backgrounds or additional elements."

        #feature_respone을 텍스트로 comfyUI에 이미지 생성 요구
        workflow_path = 'app/database/workflows/workflow_default.json'
        image = defualt_img_gen(workflow_path,feature_response)

        #생성된 이미지를 입력으로 사용

        """
        glb 파일을 불러와서  mesh_data 추출
        """
        glb_path = glb_dic[petMeshNumber]
        mesh_data = extract_mesh(glb_path)
        
        #테스트용 이미지 사용
        #image = Image.open(tmp_ful)

        bake_resolution = 1024

        texture_image = generate_texture(mesh_data, image, bake_resolution)

        texture_path = os.path.join(texture_dir,f"texure_{userId}.png")
        texture_image.save(texture_path)


        return FileResponse(texture_path,media_type='image/png',filename=f"texture_{userId}.png")
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}"
        )
    
    

@texture_router.post('/img2texture',tags=['texture'])
async def img2texture(userId:int = Form(...),petMeshNumber:int = Form(...),image:UploadFile = File(...)):

    #print(pet_request.image.content_type)
    """
    이미지를 입력 받아 반려동물 특징을 추출하고 
    특징에 맞는 동물 털 이미지 생성합니다.
    모델링 넘버를 이용하여 glb 파일을 검색합니다.
    glb 파일로 생성한 텍스처에 생성형 이미지로 동물 텍스처를 생성합니다.
    """
    try:
        """
        이미지를 입력받고 저장하는 코드
        """
        print('1')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        file_location =os.path.join(img_dir,image.filename)

        with open(file_location,'wb') as b:
            shutil.copyfileobj(image.file,b)


        

        """
        molmo 비전 모델을 이용해서 이미지 특징을 추출하고 
        openAI로 특징을 정제하는 작업
        """

        print('2')
        ### 여기 부위별 색상 특징 추출하게 바꾸고 출력하게 해서 테스트 해보기
        with open(file_location, "rb") as image_file:
            files = {'image': (image.filename, image_file, 'image/jpeg')}
            data = {"text": "Describe this image"}
            print('2-1')
            # 외부 API 호출
            response = requests.post("http://221.163.19.142:55508/chatbot/im_text", files=files)
            print('2-2')

        print('3')
        print("이미지 처리 완료")

        # 결과 출력
        print("추출된 텍스트:",response.text)
        

        # 
        '''
        response = """
        1. The overall color of the animal's fur is primarily a light brown or tan color.

        2. The fur appears to be short and smooth. It's not coarse or long.

        3. Yes, the animal has multiple colors distinctly separated into patches:
        - White chest
        - Orange back
        - Black spots

        4. Yes, there is a visible stripe or pattern on the animal's fur:
        - The pattern resembles a Tabby pattern
        - The stripes are thin and continuous, resembling a Mackerel Tabby
        - The color of the stripes or pattern is brown

        5. No, the fur is not of the same color throughout the entire body. It's a combination of the colors mentioned in question 3.

        6. Identification of fur color for specific body parts:
        - Legs: Not specified in the image description
        - Belly: Not specified in the image description
        - Chest: White
        - Back: Orange
        - Face: Not specified in the image description          
        """
        '''
        # response를 llm에 보내서 프롬프트 생성하기
        model = load_llm_model()


        ### 
        description_response = get_descriptionLLM_response(model,response.text)
        prompt_response = get_promptLLM_response(model,description_response)

        main_prompt = prompt_response.main
        leg_prompt = prompt_response.leg
        tail_prompt = prompt_response.tail
        stripe_prompt = prompt_response.stripe
        print(tuple([main_prompt,stripe_prompt]))

        if description_response.stripe_type == 'Mackerel':
            # tabby용 prompt를 사용하는 get_promptLLM_response 사용 
            print("---------------------------")
            print("Mackerel 출력")
            print(main_prompt)
            image = tabby_cat(main_prompt,stripe_prompt)
        elif description_response.stripe_type == 'Calico':
            # Calico용 prompt를 사용하는 get_promptLLM_response 사용 
            print("---------------------------")
            print("Calico 출력")
            print(main_prompt)
            image = calico_cat(main_prompt,stripe_prompt)## Calico용으로 변경하기
        
        elif description_response.stripe_type == 'Mackerel+Calico':
            # Calico용 prompt를 사용하는 get_promptLLM_response 사용 
            print("---------------------------")
            print("Mackerel+Calico 출력")
            print(main_prompt)
            print("C : ",prompt_response.calico_stripe)
            print("M : ",prompt_response.mackerel_stripe)
            image = vita_cat(main_prompt,stripe_prompt)## Calico용으로 변경하기
        else :
            # 일반 고양이용 get_promptLLM_response 사용 
            print("---------------------------")
            print("일반 고양이 출력")
            print(main_prompt)
            image = inpainting2part(main_prompt)
        
        
        """
        특징 기반 metrial 이미지 생성
        """
        
        #테스트용 특징
        #feature_respone = "Create a close-up image of animal fur with a solid blure color. The fur is glossy, sleek, without any patterns or additional colors. The texture should reflect light slightly to emphasize the smooth and shiny appearance of the black fur. There should be no visible background or additional elements, focusing solely on the animal's sleek black fur."
        #feature_respone = "Create a close-up image of animal fur with a light orange main color. The fur should have no visible patterns and be a solid color across the body. The fur should be short, smooth, and have bright but slightly muted colors for a more natural appearance"


        #임시용 feature_response
        #feature_response = "Create an image filled entirely with the natural colors of the animal's fur. The fur should be medium length, moderately thick, and possess a subtle sheen that catches the light. The primary color should be a natural blend of deep, dark brown with subtle hints of rich, golden undertones. This combination should be smooth and seamlessly integrated, creating a harmonious color palette. The image should maintain perfect symmetry from left to right, focusing solely on these specified fur colors without any backgrounds or additional elements."

        #feature_respone을 텍스트로 comfyUI에 이미지 생성 요구
        
        

        #생성된 이미지를 입력으로 사용

        """
        glb 파일을 불러와서  mesh_data 추출
        """

        texture_path = os.path.join(texture_dir,f"texure_{userId}.png")
        image.save(texture_path)


        return FileResponse(texture_path,media_type='image/png',filename=f"texture_{userId}.png")
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}"
        )
        
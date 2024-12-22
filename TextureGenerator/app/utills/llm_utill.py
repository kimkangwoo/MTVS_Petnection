from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field


class FurFeatures(BaseModel):
    main:str = Field(description="전체적인 털 색상을 한 색상으로 설명하시오.이때 줄무늬 색상은 색상조합에서 제외하세요")
    stripe:str = Field(description="줄무늬의 색상을 한 색상으로만 설명하시오.")
    feature:str = Field(description="털의 질감,길이등 털의 특징을 설명하시오.")
    leg:str = Field(description="다리의 색상을 한가지 색상으로 설명하시오.")
    chest:str = Field(description="가슴에 색상을 한가지 색상으로 설명하시오")
    tail:str = Field(description="꼬리에 색상을 한가지 색상으로 설명하시오")
    stripe_type:str = Field(description="줄무늬의 종류를 [NoStripe/Mackerel/Calico or Bicolor/Mackerel+Calico] 4가지 중 하나로 표기하시오.")
    animal_type:str = Field(description="동물의 종류를 한 단어로 작성하시오.")

class GenerationPrompt(BaseModel):
    main:str = Field(description="전체적인 털 색상을 한 색상에 털로만 가득찬 이미지를 생성하는 프롬프트를 작성하세요.")
    stripe:str = Field(description="줄무늬의 색상을 줄무늬 패턴을 명시적으로 언급하지 않고 한 색상으로 자연스러운 털로 가득찬 이미지를 질감이 약하게 생성하는 프롬프트를 작성하세요. ")
    leg:str = Field(description="다리의 색상을 한 색상에 털 이미지를 털 이미지를 생성하는 프롬프트를 작성하세요")
    tail:str = Field(description="꼬리의 색상을 한 색상에 털 이미지를 생성하는 프롬프트를 작성하세요")
    calico_stripe:str = Field(description="Calico 특징 중 spoted(점 무늬) 색상에 털로 가득찬 이미지를 생성하는 프롬프트를 작성하세요. 털의 상세한 패턴(spoted,the mackerel pattern)는 프롬프트에 명시하지 마시오.")
    mackerel_stripe:str = Field(description="Mackerel 특징으로 설명한 색상에 털로 가득찬 이미지를 생성하는 프롬프트를 작성하세요.털의 상세한 패턴(spoted,the mackerel pattern)는 프롬프트에 명시하지 마시오.")


output_parser = PydanticOutputParser(pydantic_object=FurFeatures)
format_instructions = output_parser.get_format_instructions()

generation_output_parser = PydanticOutputParser(pydantic_object=GenerationPrompt)
generation_format_instruction =generation_output_parser.get_format_instructions()

prompt = PromptTemplate.from_template(
    """
당신은 DESCIPTION에 설명된 특징을 FORMAT에 맞게 나누어서 설명하는 어시스턴트입니다.

만약 색상이 "주황색" 또는 "오렌지색"으로 표현되었다면, 반드시 이를 "밝은 황갈색"으로 변경하여 작성하시오. 이 변경은 예외 없이 적용하시오.
만약 무늬가 있을 경우 검은색이 포함되었다면 검은 색은 Dark Chocolate Brown으로 변경하여 작성하시오.
만약 메인 색상이 brown이고 부위에 어두운 계열 색상과 조합되면 brown을 Tawny Brown으로 변경하여 작성하시오.
만약 메인 색상이 brown이고 부위에 밝은 계열 색상과 조합되면 brown을 Sandy Brown으로 변경하여 작성하시오.
만약 메인 색상이 brown이고 부위에 밝은 계열 색상과 어두운 계열 색상이 조합되지 않으면 그대로 작성하시오.


줄무늬 색상은 전체색과 자연스럽게 대조되도록 작성하되, 주어진 색상(예: dark gray)은 반드시 유지하시오. 
다른 색상으로 축약하거나 단순화하지 마시오.

FORMAT은 아래와 같이 작성되어야 하며, 모든 필드는 정확히 채워져야 합니다. 필드가 없으면 "N/A"로 표시하시오.


QUESTION:
{question}

DESCRIPTION:
{description}

FORMAT:
{format}
 
"""
)

prompt = prompt.partial(format=format_instructions)


gen_prompt = PromptTemplate.from_template(
    """
당신은 이미지 프롬프트를 작성하는 어시스턴트입니다.
아래 주의사항을 참고하여 DESCIPTION에 설명된 동물의 부위별 털 특징을 이용하여 부위별로 프롬프트를 FORMAT에 맞게 작성하시오.

1. 이미지를 생성할 때 털의 질감이 뚜렷하게 나오게 작성하시오.
2. 모든 이미지 생성문구에는 Symmetrical composition, perfectly symmetrical, Evenly mirrored, balanced symmetry 와 같은 좌우가 대칭되게 생성하는 문구를 하나는 반드시 포함하시오.
3. 이미지는 동물의 털을 생성한다고 명시하시오.
4. FORMAT에는 부위(belly, paws, and face areas)나 동물의 형상을 명시적으로 언급하지 마시오.
5. FORMAT의 모든 프롬프트는 동물의 털로 가득 찬 화면을 생성하도록 작성하시오.
6. 특정 동물의 종류를 연상시키는 표현(feline elegance 등)을 사용하지 마시오.
7. 모든 프롬프트는 간결하고 명료한 구문으로 작성하시오.
8. 모든 프롬프트는 털 이미지를 생성하시오.
9. 모든 프롬프트는 영어로 생성하시오.

FORMAT은 아래와 같이 작성되어야 하며, 모든 필드는 정확히 채워져야 합니다. 필드가 없으면 "N/A"로 표시하시오.

QUESTION:
{question}

DESCRIPTION:
{description}

FORMAT:
{format}

"""
)

gen_prompt = gen_prompt.partial(format=generation_format_instruction)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def load_llm_model(model_name="gpt-4o-2024-08-06"):
    """LLM 모델을 로드합니다."""
    model = ChatOpenAI(model_name=model_name)
    return model

def prepare_prompt_template(system_prompt):
    """프롬프트 템플릿을 준비합니다."""
    custom_prompt = ChatPromptTemplate.from_template(system_prompt)
    return custom_prompt

def format_prompt(custom_prompt, **kwargs):
    """주어진 변수로 프롬프트를 포맷팅합니다."""
    formatted_prompt = custom_prompt.format(**kwargs)
    return formatted_prompt

'''
        -----사용법-----
formatted_prompt = format_prompt(
    custom_prompt,
    history=conversation_history,
    context=context,
    text=text
)
'''

def get_llm_response(model, custom_prompt, formatted_prompt, retriever=None,image_url=None):

    if retriever:
        input_data = {"context": retriever, "text": RunnablePassthrough()}
    else:
        input_data = {"context": RunnablePassthrough(),"text": RunnablePassthrough()}

    if image_url:
        input_data["image_url"] = RunnablePassthrough()

    rag_chain = (
        input_data
         | custom_prompt
         | model
         | StrOutputParser()
    )
    #print(f"Type of formatted_prompt: {type(formatted_prompt)}")
    #rag_chain = model | StrOutputParser()

    #print(rag_chain)
    response = rag_chain.invoke(formatted_prompt)
    return response


'''
-----사용법-----
response = get_llm_response(model,  formatted_prompt, openai_retriever,)
'''

def get_descriptionLLM_response(model,description):
    chain = prompt|model|output_parser

    response = chain.invoke(
        {
            "question":"아래 내용을 분류해주세요",
            "description":description
        }
    )

    return response

def get_promptLLM_response(model,description):
    chain = gen_prompt|model|generation_output_parser

    response = chain.invoke(
        {
            "question":"해당 내용으로 프롬프트를 작성해주세요.",
            "description":description
        }
    )

    return response
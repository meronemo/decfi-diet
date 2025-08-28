from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Literal, List
import json
import re
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://clovastudio.stream.ntruss.com/v1/openai"
)

Intent = Literal[
    "INCLUDE_ITEM", # 1. 특정 음식 포함
    "EXCLUDE_ITEM", # 2. 특정 음식 제외
    "NUTRIENT", # 3. 영양소 목표치 설정
    "PREFERENCE", # 4. 선호(취향, 맵기 등)
]

Strength = Literal["soft", "hard"] # 선호 / 강제

Nutrient = Literal["calorie", "protein", "fat", "carbon", "sodium", "sugar", "fiber"]
# 칼로리, 단백질, 지방, 탄수화물, 나트륨, 당, 식이섬유

Bound = Literal["lower", "greater", "equal"]

PreferenceType = Literal[
    "spice_level", # 맵기
    # "cooking_method", # 조리법(튀김, 구이, 찜 등)
    "food_group", # 샐러드/육류/해산물/곡류 등
]
SpiceLevel = Literal["none", "low", "medium", "high"]
# CookingMethod = Literal["boiled", "steamed", "grilled", "baked", "fried", "raw"]
FoodGroup = Literal[""]

class Constraints(BaseModel):
    intent: Intent # 제약 종류
    strength: Strength = "soft" # 제약 강도
    food_item: Optional[str] = None # (1, 2) 음식 이름
    nutrient: Optional[Nutrient] = None # (3) 영양소 종류
    bound_type: Optional[Bound] = None # (3) 영양소 제약 종류
    bound_value: Optional[float] = None # (3) 영양소 제약 값
    preference_type: Optional[PreferenceType] = None # (4) 선호도 제약 종류
    spice_level: Optional[SpiceLevel] = None # (4) 맵기 선호
    # cooking_method: Optional[CookingMethod] = None # (4) 조리법 선호
    food_group: Optional[str] = None # (4) 종류 선호


def parse_constraints(user_text: str) -> List[Constraints]:
    system_prompt = """
        너는 사용자의 자연어 문장을 기반으로 식단 제약 조건을 JSON으로 변환하는 파서이다.  
        출력은 반드시 {"constraints": [...]} 형태의 JSON 하나만 생성해야 한다.  
        JSON 외의 설명, 텍스트는 절대 포함하지 않는다.

        스키마 정의:
        - intent: 문자열 (반드시 아래 중 하나)
          "INCLUDE_ITEM": 특정 음식 반드시 포함
          "EXCLUDE_ITEM": 특정 음식 반드시 제외
          "NUTRIENT": 영양소 목표치 설정
          "PREFERENCE": 선호(취향, 맵기, 음식 그룹 등)
        - strength: 문자열 (반드시 아래 중 하나)
          "hard": 반드시 지켜야 하는 강제 조건
          "soft": 선호 사항
          기본값은 "soft"
        - food_item: 문자열 | null 
          intent가 INCLUDE_ITEM 또는 EXCLUDE_ITEM일 때 사용
        - nutrient: 문자열 | null (반드시 아래 중 하나)
          calorie | protein | fat | carbon | sodium | sugar | fiber
          각각 칼로리, 단백질, 지방, 탄수화물, 나트륨, 당, 식이섬유임
          intent가 NUTRIENT일 때 사용
        - bound_type: 문자열 | null (반드시 아래 중 하나)
          lower | greater | equal
          영양소의 값이 bound_value보다 적어야 / 커야 / 같아야 하는지를 나타냄
          intent가 NUTRIENT일 때 사용
        - bound_value: 숫자 | null  
          영양소 제약 값
          intent가 NUTRIENT일 때 사용
        - preference_type: 문자열 | null (반드시 아래 중 하나)
          spice_level | food_group
          intent가 PREFERENCE일 때 사용
        - spice_level: 문자열 | null (반드시 아래 중 하나)
          none | low | medium | high
          preference_type이 spice_level일 때 사용
        - food_group: 문자열 | null
          샐러드, 육류, 해산물, 곡류 등  
          preference_type이 food_group일 때 사용

        규칙:
        1. 여러 제약 조건이 동시에 등장하면 constraints 배열에 각각 추가한다.
        2. 불명확한 경우 strength="soft"로 설정한다.
        3. 출력은 반드시 유효한 JSON이어야 하며, JSON 외 다른 텍스트를 전혀 포함하지 않는다.
    """

    response = client.chat.completions.create(
        model="HCX-005",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.0
    )

    res = response.choices[0].message.content
    
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", res, re.IGNORECASE)
    if fence:
        res = fence.group(1).strip()
    else:
        res = res.strip()

    try:
        js = json.loads(res)
    except Exception as e:
        print("JSON 파싱 중 오류", e)

    return [Constraints(**c) for c in js["constraints"]]


if __name__ == "__main__":
    user_input = "오늘은 맵지 않게, 육류 위주로, 1800kcal보다 적도록 추천해줘. 또, 이건 필수는 아닌데 불고기가 들어갔으면 좋겠어."
    constraints = parse_constraints(user_input)
    for c in constraints:
        print(c)

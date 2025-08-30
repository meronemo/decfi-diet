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

FoodItem = Literal["김치찌개", "된장찌개", "비빔밥", "불고기", "삼겹살", "갈비탕", "설렁탕", "냉면", "칼국수", "떡볶이", "순대국", "제육볶음", "잡채", "김밥", "참치김밥", "라면", "짜장면", "짬뽕", "볶음밥", "김치볶음밥", "피자", "햄버거", "치킨", "후라이드치킨", "양념치킨", "돈까스", "카레라이스", "오므라이스", "스파게티", "토마토파스타", "크림파스타", "스테이크", "샐러드", "시저샐러드", "그릭요거트", "과일샐러드", "바나나", "사과", "배", "딸기", "포도", "수박", "멜론", "망고", "파인애플", "오렌지", "레몬", "자몽", "블루베리", "체리", "복숭아", "참외", "토마토", "가지", "오이", "양배추", "브로콜리", "시금치", "감자", "고구마", "옥수수", "콩밥", "현미밥", "흰쌀밥", "잡곡밥", "죽", "미역국", "콩나물국", "계란찜", "계란후라이", "계란말이", "메추리알조림", "두부조림", "연두부", "순두부찌개", "고등어구이", "삼치구이", "갈치조림", "오징어볶음", "낙지볶음", "새우볶음", "조개탕", "홍합탕", "전복죽", "게장", "젓갈", "김치", "깍두기", "총각김치", "열무김치", "파김치", "백김치", "깻잎장아찌", "마늘장아찌", "고추장아찌", "청국장", "감자탕", "해장국", "추어탕", "육개장", "우동", "쌀국수", "포케", "타코", "부리토", "핫도그", "샌드위치", "크로와상", "베이글", "식빵", "호밀빵", "치즈케이크", "티라미수", "초콜릿케이크", "도넛", "머핀", "크루아상샌드위치", "바게트", "프렌치토스트", "팬케이크", "와플", "쿠키", "프렌치프라이", "어니언링", "치즈스틱", "모짜렐라피자", "페퍼로니피자", "마르게리타피자", "고르곤졸라피자", "씨푸드피자", "연어스테이크", "연어샐러드", "훈제연어", "베이컨", "소시지", "햄", "프라이드포테이토", "스프링롤", "춘권", "군만두", "찐만두", "물만두", "김치만두", "고기만두", "야채만두", "샤브샤브", "훠궈", "퐁듀", "라자냐", "리조또", "샤클슈카", "팔라펠", "후무스", "케밥", "샤와르마", "쿠스쿠스", "카프레제샐러드", "리코타샐러드", "양상추샐러드", "감자샐러드", "단호박죽", "팥죽", "녹두죽", "호박죽", "닭죽", "삼계탕", "닭칼국수", "닭볶음탕", "닭강정", "치킨커틀릿", "연어초밥", "광어초밥", "참치초밥", "새우초밥", "계란초밥", "장어덮밥", "카츠동", "규동", "텐동", "가츠동", "소고기덮밥", "치킨마요덮밥", "연어마요덮밥", "스시롤", "우나기롤", "드래곤롤", "캘리포니아롤", "참치롤", "새우롤", "연어롤", "초밥세트", "스시세트", "돈코츠라멘", "미소라멘", "시오라멘", "스파이시라멘", "파스타샐러드", "크루통샐러드", "토마토스프", "양송이스프"]

Nutrient = Literal["calorie", "protein", "fat", "carbon", "sodium", "sugar", "fiber"]
# 칼로리, 단백질, 지방, 탄수화물, 나트륨, 당, 식이섬유

Bound = Literal["lower", "greater", "equal"]

PreferenceType = Literal[
    "spice_level", # 맵기
    "food_group", # 샐러드/육류/해산물/곡류 등
]
SpiceLevel = Literal["low", "medium", "high"]
FoodGroup = Literal["soup", "rice", "meat", "noodle", "salad", "fruit_veg", "seafood", "side_ferment", "bread_dessert", "fried_snack", "sushi_roll", "etc"]
# 국/찌개/스프, 밥류, 육류, 면류, 샐러드, 과일/채소, 해산물, 반찬/발효, 빵/디저트, 튀김/간식, 초밥/롤, 기타

class Constraints(BaseModel):
    intent: Intent # 제약 종류
    strength: Strength = "soft" # 제약 강도
    food_item: Optional[FoodItem] = None # (1, 2) 음식 이름
    nutrient: Optional[Nutrient] = None # (3) 영양소 종류
    bound_type: Optional[Bound] = None # (3) 영양소 제약 종류
    bound_value: Optional[float] = None # (3) 영양소 제약 값
    preference_type: Optional[PreferenceType] = None # (4) 선호도 제약 종류
    spice_level: Optional[SpiceLevel] = None # (4) 맵기 선호
    food_group: Optional[FoodGroup] = None # (4) 종류 선호

def parse_constraints(user_text: str) -> List[Constraints]:
    system_prompt = """
        너는 사용자의 자연어 문장을 기반으로 식단 제약 조건을 JSON으로 변환하는 파서이다.  
        출력은 반드시 {"constraints": [...]} 형태의 JSON 하나만 생성해야 한다.  
        JSON 외의 설명, 텍스트는 절대 포함하지 않는다.
        여러 제약 조건이 동시에 등장하면 constraints 배열에 각각 추가한다.
        각 제약의 강도가 분명히 나타나는 경우 외에는 strength="soft"로 설정한다.

        스키마 정의:
        - intent: 문자열 (반드시 아래 중 하나)
          "INCLUDE_ITEM": 특정 음식 반드시 포함
          "EXCLUDE_ITEM": 특정 음식 반드시 제외
          "NUTRIENT": 영양소 목표치 설정
          "PREFERENCE": 선호(취향, 맵기, 음식 종류 등)
          INCLUDE_ITEM, EXCLUDE_ITEM은 특정 음식의 이름이 명확히 나타난 경우에만 사용
          특정 음식 종류에 대한 선호가 나타난 경우에는 PREFERENCE 사용
        - strength: 문자열 (반드시 아래 중 하나)
          "hard": 반드시 지켜야 하는 강제 조건
          "soft": 선호 사항
          기본값은 "soft"
        - food_item: 문자열 | null (반드시 아래 중 하나)
          아래 제시된 음식 이름 중에서만 작성, 사용자가 작성한 음식이 존재하지 않는 경우 현저히 차이가 나지 않는 경우에만 아래 제시된 것 중 가장 가까운 것으로 대체
          김치찌개 | 된장찌개 | 비빔밥 | 불고기 | 삼겹살 | 갈비탕 | 설렁탕 | 냉면 | 칼국수 | 떡볶이 | 순대국 | 제육볶음 | 잡채 | 김밥 | 참치김밥 | 라면 | 짜장면 | 짬뽕 | 볶음밥 | 김치볶음밥 | 피자 | 햄버거 | 치킨 | 후라이드치킨 | 양념치킨 | 돈까스 | 카레라이스 | 오므라이스 | 스파게티 | 토마토파스타 | 크림파스타 | 스테이크 | 샐러드 | 시저샐러드 | 그릭요거트 | 과일샐러드 | 바나나 | 사과 | 배 | 딸기 | 포도 | 수박 | 멜론 | 망고 | 파인애플 | 오렌지 | 레몬 | 자몽 | 블루베리 | 체리 | 복숭아 | 참외 | 토마토 | 가지 | 오이 | 양배추 | 브로콜리 | 시금치 | 감자 | 고구마 | 옥수수 | 콩밥 | 현미밥 | 흰쌀밥 | 잡곡밥 | 죽 | 미역국 | 콩나물국 | 계란찜 | 계란후라이 | 계란말이 | 메추리알조림 | 두부조림 | 연두부 | 순두부찌개 | 고등어구이 | 삼치구이 | 갈치조림 | 오징어볶음 | 낙지볶음 | 새우볶음 | 조개탕 | 홍합탕 | 전복죽 | 게장 | 젓갈 | 김치 | 깍두기 | 총각김치 | 열무김치 | 파김치 | 백김치 | 깻잎장아찌 | 마늘장아찌 | 고추장아찌 | 청국장 | 감자탕 | 해장국 | 추어탕 | 육개장 | 우동 | 쌀국수 | 포케 | 타코 | 부리토 | 핫도그 | 샌드위치 | 크로와상 | 베이글 | 식빵 | 호밀빵 | 치즈케이크 | 티라미수 | 초콜릿케이크 | 도넛 | 머핀 | 크루아상샌드위치 | 바게트 | 프렌치토스트 | 팬케이크 | 와플 | 쿠키 | 프렌치프라이 | 어니언링 | 치즈스틱 | 모짜렐라피자 | 페퍼로니피자 | 마르게리타피자 | 고르곤졸라피자 | 씨푸드피자 | 연어스테이크 | 연어샐러드 | 훈제연어 | 베이컨 | 소시지 | 햄 | 프라이드포테이토 | 스프링롤 | 춘권 | 군만두 | 찐만두 | 물만두 | 김치만두 | 고기만두 | 야채만두 | 샤브샤브 | 훠궈 | 퐁듀 | 라자냐 | 리조또 | 샤클슈카 | 팔라펠 | 후무스 | 케밥 | 샤와르마 | 쿠스쿠스 | 카프레제샐러드 | 리코타샐러드 | 양상추샐러드 | 감자샐러드 | 단호박죽 | 팥죽 | 녹두죽 | 호박죽 | 닭죽 | 삼계탕 | 닭칼국수 | 닭볶음탕 | 닭강정 | 치킨커틀릿 | 연어초밥 | 광어초밥 | 참치초밥 | 새우초밥 | 계란초밥 | 장어덮밥 | 카츠동 | 규동 | 텐동 | 가츠동 | 소고기덮밥 | 치킨마요덮밥 | 연어마요덮밥 | 스시롤 | 우나기롤 | 드래곤롤 | 캘리포니아롤 | 참치롤 | 새우롤 | 연어롤 | 초밥세트 | 스시세트 | 돈코츠라멘 | 미소라멘 | 시오라멘 | 스파이시라멘 | 파스타샐러드 | 크루통샐러드 | 토마토스프 | 양송이스프
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
          맵기 수준에 대한 선호도가 나타나면 spice_level
          특정 음식 종류에 대한 선호도가 나타나면 food_group
          intent가 PREFERENCE일 때 사용
        - spice_level: 문자열 | null (반드시 아래 중 하나)
          low | medium | high
          preference_type이 spice_level일 때 사용
        - food_group: 문자열 | null (반드시 아래 중 하나)
          soup | rice | meat | noodle | salad | fruit_veg | seafood | side_ferment | bread_dessert | fried_snack | sushi_roll | etc
          국/찌개/스프, 밥류, 육류, 면류, 샐러드, 과일/채소, 해산물, 반찬/발효, 빵/디저트, 튀김/간식, 초밥/롤, 기타
          preference_type이 food_group일 때 사용
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
        return [Constraints(**c) for c in js["constraints"]]
    except Exception as e:
        print("JSON 파싱 중 오류", e)
        return js

if __name__ == "__main__":
    user_input = "오늘은 맵지 않게, 육류 위주로, 1800kcal보다 적도록 추천해줘. 그리고 불고기가 들어갔으면 좋겠어."
    constraints = parse_constraints(user_input)
    for c in constraints:
        print(c)

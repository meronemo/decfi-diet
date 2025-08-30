import pandas as pd
from pulp import *
from fastapi import APIRouter

class MealRecommendation:
    def __init__(self):
        # 음식 데이터 로드
        self.food_data = pd.read_csv('food_data.csv')
    
    def calculate_daily_calories(self, weight: float, height: float, age: int, gender: str, activity: int, goal: str) -> float:
        # 기초대사량 계산 (Mifflin St. Jeor)
        s = 5 if gender == "male" else -161
        bmr = 10 * weight + 6.25 * height - 5 * age + s
        
        # 일일 권장 칼로리 계산
        activity_factor_map = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
        activity_factor = activity_factor_map[activity]
        goal_multipliers = {
            'maintain': 1.0,
            'loss': 0.85, # 기초대사량 85%
            'gain': 1.15 # 기초대사량 115%
        }
        goal_multiplier = goal_multipliers[goal]
        
        daily_calories = bmr * activity_factor * goal_multiplier
        return daily_calories
    
    def calculate_meal_targets(self, daily_calories: float) -> dict:
        # 한 끼 식사의 영양소 목표값 계산 (일일 권장량의 40%)
        meal_calories = daily_calories * 0.4
        
        carb_ratio = 0.60 # 탄수화물 60%
        protein_ratio = 0.15 # 단백질 15%
        fat_ratio = 0.25 # 지방 25%
        
        targets = {
            'calories': meal_calories,
            'carbs': (meal_calories * carb_ratio) / 4,
            'protein': (meal_calories * protein_ratio) / 4,
            'fat': (meal_calories * fat_ratio) / 9,
            'fiber': max(8, meal_calories / 100), # 최소 8g, 칼로리 100당 1g
            'sodium': 800, # mg, 일일 권장량의 40%
            'sugar': meal_calories * 0.1 / 4 # 칼로리의 10% 이하
        }
        
        return targets
    
    def solve_meal_optimization(self, targets: dict) -> list:
        n_foods = len(self.food_data)
        
        # 의사결정 변수: 각 음식의 선택 여부
        food_vars = [LpVariable(f"food_{i}", cat='Binary') for i in range(n_foods)]
        
        prob = LpProblem("meal_recommendation", LpMinimize)
        
        # 편차 변수
        cal_pos = LpVariable("cal_pos", lowBound=0)
        cal_neg = LpVariable("cal_neg", lowBound=0)
        protein_pos = LpVariable("protein_pos", lowBound=0)
        protein_neg = LpVariable("protein_neg", lowBound=0)
        carb_pos = LpVariable("carb_pos", lowBound=0)
        carb_neg = LpVariable("carb_neg", lowBound=0)
        fat_pos = LpVariable("fat_pos", lowBound=0)
        fat_neg = LpVariable("fat_neg", lowBound=0)
        
        # 목적 함수: 가중치 적용 편차 최소화
        prob += (1 * (cal_pos + cal_neg) +
                1.5 * (protein_pos + protein_neg) +
                0.5 * (carb_pos + carb_neg) +
                0.5 * (fat_pos + fat_neg))
        
        # 영양소 제약 조건
        prob += (lpSum([self.food_data.iloc[i]['칼로리'] * food_vars[i] 
                       for i in range(n_foods)]) - targets['calories'] 
                == cal_pos - cal_neg)
        
        prob += (lpSum([self.food_data.iloc[i]['단백질'] * food_vars[i] 
                       for i in range(n_foods)]) - targets['protein'] 
                == protein_pos - protein_neg)
        
        prob += (lpSum([self.food_data.iloc[i]['탄수화물'] * food_vars[i] 
                       for i in range(n_foods)]) - targets['carbs'] 
                == carb_pos - carb_neg)
        
        prob += (lpSum([self.food_data.iloc[i]['지방'] * food_vars[i] 
                       for i in range(n_foods)]) - targets['fat'] 
                == fat_pos - fat_neg)
        
        # 총 음식 개수 제한 (4-8개)
        prob += lpSum(food_vars) >= 4
        prob += lpSum(food_vars) <= 8
        
        # 주식(밥류, 면류, 초밥/롤) 최소 1개
        main_dishes = [i for i, row in self.food_data.iterrows() 
                        if row['종류'] in ['밥류', '면류', '초밥/롤']]
        if main_dishes:
            prob += lpSum([food_vars[i] for i in main_dishes]) == 1
        
        # 과일/채소, 샐러드, 디저트/간식 각각 최대 1개
        fruits_veggies = [i for i, row in self.food_data.iterrows() 
                            if row['종류'] == '과일/채소']
        if fruits_veggies:
            prob += lpSum([food_vars[i] for i in fruits_veggies]) <= 1

        salad = [i for i, row in self.food_data.iterrows() 
                            if row['종류'] == '샐러드']
        if salad:
            prob += lpSum([food_vars[i] for i in salad]) <= 1

        dessert_snacks = [i for i, row in self.food_data.iterrows() 
                            if row['종류'] in ['빵/디저트', '튀김/간식']]
        if dessert_snacks:
            prob += lpSum([food_vars[i] for i in dessert_snacks]) <= 1
        
        # 칼로리 범위 제한 (목표 20% 내외)
        min_calories = targets['calories'] * 0.8
        max_calories = targets['calories'] * 1.2
        prob += lpSum([self.food_data.iloc[i]['칼로리'] * food_vars[i] 
                        for i in range(n_foods)]) >= min_calories
        prob += lpSum([self.food_data.iloc[i]['칼로리'] * food_vars[i] 
                        for i in range(n_foods)]) <= max_calories
        
        # 나트륨 범위 제한 (목표 120% 내외)
        prob += lpSum([self.food_data.iloc[i]['나트륨'] * food_vars[i] 
                        for i in range(n_foods)]) <= targets['sodium'] * 1.2
        
        # 당분 범위 제한 (목표 120% 내외)
        prob += lpSum([self.food_data.iloc[i]['당'] * food_vars[i] 
                        for i in range(n_foods)]) <= targets['sugar'] * 1.2

        prob.solve(PULP_CBC_CMD(msg=0))
        
        if prob.status == 1:
            selected_foods = []
            for i in range(n_foods):
                if food_vars[i].varValue == 1:
                    food_name = self.food_data.iloc[i]['음식명']
                    food_info = {
                        'name': food_name,
                        'category': self.food_data.iloc[i]['종류'],
                        'kcal': float(self.food_data.iloc[i]['칼로리']),
                        'protein_g': float(self.food_data.iloc[i]['단백질']),
                        'fat_g': float(self.food_data.iloc[i]['지방']),
                        'carb_g': float(self.food_data.iloc[i]['탄수화물']),
                        'sodium_mg': float(self.food_data.iloc[i]['나트륨']),
                        'sugar_g': float(self.food_data.iloc[i]['당'])
                    }
                    selected_foods.append(food_info)
            return selected_foods
        else:
            status_msg = {
                -1: "최적해를 찾을 수 없음",
                -2: "제약 조건에 오류가 있음", 
                -3: "문제가 unbounded임"
            }
            print(f"문제 해결 실패: {status_msg.get(prob.status, '알 수 없는 오류')}")
            return None
    
    def calculate_totals(self, selected_foods: list) -> dict:
        # 선택된 음식들의 영양소 총합 계산
        if not selected_foods:
            return None
        
        totals = {
            'kcal': 0.0,
            'protein_g': 0.0,
            'fat_g': 0.0,
            'carb_g': 0.0,
            'sodium_mg': 0.0,
            'sugar_g': 0.0
        }
        
        for food in selected_foods:
            totals['kcal'] += food['kcal']
            totals['protein_g'] += food['protein_g']
            totals['fat_g'] += food['fat_g']
            totals['carb_g'] += food['carb_g']
            totals['sodium_mg'] += food['sodium_mg']
            totals['sugar_g'] += food['sugar_g']

        for key in totals:
            totals[key] = round(totals[key], 1)
        
        return totals

router = APIRouter()
@router.get('/')
def recommend_one_meal(height: float, weight: float, age: int, gender: str, activity: int, goal: str):
    """
    사용자 정보를 바탕으로 한 끼 식사 추천
    
    Args:
        height: 키 (cm)
        weight: 몸무게 (kg)  
        age: 나이
        gender: 성별 ('male' 또는 'female')
        activity: 활동 수준 (1-5)
        goal: 목표 ('maintain', 'loss', 'gain')
    
    Returns:
        dict: 추천 식단과 영양소 총합 정보
    """
    meal_recommender = MealRecommendation()
    try:
        daily_calories = meal_recommender.calculate_daily_calories(weight, height, age, gender, activity, goal)
        meal_targets = meal_recommender.calculate_meal_targets(daily_calories)
        selected_foods = meal_recommender.solve_meal_optimization(meal_targets)
        
        if selected_foods is None:
            return {
                "status": "fail"
            }
        
        totals = meal_recommender.calculate_totals(selected_foods)
        
        return {
            "status": "success",
            "items": selected_foods,
            "totals": totals
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"추천 과정에서 오류가 발생했습니다: {str(e)}",
            "error_details": str(e)
        }
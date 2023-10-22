from .BMR import BMR
# from Food_Nutrition_checker import food_nutrition_checker

def get_calorie_goal(gender, age, height, weight, exec_lvl, weight_goal, user_meal_type:str, **kwargs):
    tdee = BMR(gender, age, height, weight, exec_lvl, weight_goal)

    # while True:
        # user_meal_type  = input("Enter meal to customize (breakfast/lunch/dinner): ").lower()
        
        # if user_meal_type  in ['breakfast', 'lunch', 'dinner']:
        #     break
        # else:
        #     print("Invalid meal type. Please choose breakfast, lunch, or dinner.")
    assert user_meal_type in ['breakfast', 'lunch', 'dinner']
    calorie_goal = round(0.4 * tdee) if user_meal_type == 'lunch' else round(0.3 * tdee)
    return calorie_goal
    
    
# def customize_meal_plan(selected_item:dict, remaining_calories):
#     remaining_calories -= round(selected_item['nutrition_values']['Energy_(kcal)'])

#     if remaining_calories > 0:
#         print(f"You have {remaining_calories} kcal of calories remaining for your {user_meal_type} meal plan.")
#     else:
#         print(f"You have exceeded your calorie goals for your {user_meal_type} meal plan by {-remaining_calories}.")

#     meal_plan[f'Item_{i}'] = selected_item
#     i += 1

#     add = input(f"To continue adding food item to {user_meal_type} meal plan (yes/no): ").lower()
#     if add != 'yes':
            
#     # print("\nYour customised meal plan:")
#     for key, item in meal_plan.items():
#         print(f"\n{key}: {item['selected_item']}")
#         print(f"Calories: {item['nutrition_values']['Energy_(kcal)']}")
    
#     print(f"Your calories goal is {calorie_goal} kcal and for your meal plan, your remaining calories is {remaining_calories}.")
#     return meal_plan

# if __name__ == "__main__":
    # customize_meal_plan()
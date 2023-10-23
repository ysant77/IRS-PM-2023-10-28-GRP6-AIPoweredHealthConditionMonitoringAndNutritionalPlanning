from .BMR import BMR

from django.apps import apps

food_data = apps.get_app_config("diagnosis").food_df_for_checker

def search_food_data(keyword) -> tuple:
    keyword = keyword.lower()
    # using `contain` method to search
    matching_items = food_data[food_data['name'].str.lower().str.contains(keyword, na=False)]

    if matching_items.empty: # Not found
        return None, False
    else:
        return matching_items, True

def food_nutrition_checker(matching_items, confirmation) -> dict:
    # keyword = input("Enter a keyword to search for a food item (type 'exit' to quit): ")

    # confirmation = input("Enter the number corresponding to the item you want to check, or type 'no' to search again: ")
    if confirmation.lower() == -1:
            return False
    selected_item = matching_items.iloc[int(confirmation) - 1]

    # Return a dictionary with relevant information
    return {
        'selected_item': selected_item['name'],
        'nutrition_values': selected_item.drop('name').to_dict(),
    }

def get_calorie_goal(gender, age, height, weight, exec_lvl, weight_goal, user_meal_type:str, **kwargs):
    tdee = BMR(gender, age, height, weight, exec_lvl, weight_goal)
        
    assert user_meal_type in ['breakfast', 'lunch', 'dinner']
    calorie_goal = round(0.4 * tdee) if user_meal_type == 'lunch' else round(0.3 * tdee)
    return calorie_goal
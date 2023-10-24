"""
    BMR: Basal Metabolic Rate
"""

def calculate_bmr(sex, age, height_cm, weight_kg):
    if sex.lower() == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif sex.lower() == 'female':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        raise ValueError("Invalid value for sex. Please enter 'male' or 'female'.")
    return bmr

def BMR(gender, age, height, weight, exec_lvl, weight_goal, **kwargs):
    sex = gender
    height_cm = float(height)
    weight_kg = float(weight)
    activity_level = exec_lvl

    # Calculate BMR
    bmr = calculate_bmr(sex, age, height_cm, weight_kg)
    # print(f"Your Basal Metabolic Rate (BMR) is: {bmr} calories per day")

    # print("\nSelect your activity level:")
    # print("1. Little to no exercise")
    # print("2. Light exercise (1-3 days per week)")
    # print("3. Moderate exercise (3-5 days per week)")
    # print("4. Heavy exercise (6-7 days per week)")
    # print("5. Extremely active (twice per day, extra heavy workouts)")
    
    activity_factors = {
        1: 1.2,
        2: 1.375,
        3: 1.55,
        4: 1.725,
        5: 1.9
    }

    if activity_level in activity_factors.keys():
        activity_factor = activity_factors[activity_level]
        tdee = round(bmr * activity_factor)
        # print(f"Your Total Daily Energy Expenditure (TDEE) is: {tdee} calories per day")

        # print("\nSelect your weight goal:")
        # print("1. Gain weight")
        # print("2. Lose weight")
        # print("3. Maintain weight")

        # weight_goal = input("Enter the number corresponding to your weight goal: ")

        if weight_goal == 3:
            tdee += 500
        elif weight_goal == 1:
            tdee -= 500
        elif weight_goal == 2:
            pass
        else:
            raise Exception('Wrong weight goal')
    else:
        raise Exception('Wrong exec lvl')
        
    return tdee

if __name__ == "__main__":
    BMR()
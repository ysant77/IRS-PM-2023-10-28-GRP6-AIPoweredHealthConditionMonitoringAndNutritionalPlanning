import random

from django.apps import apps

from .BMR import BMR

food_data = apps.get_app_config("diagnosis").food_df_for_ga
disease_filter_data = apps.get_app_config("diagnosis").disease_filter

def filter_data(is_vegan=False, is_halal=False, no_beef=False, low_fat=False,
                low_carb=False, high_carb=False, low_acid=False, high_protein=False,
                easily_digestible=False, high_fiber=False):
    global food_data
    # Create fruits dataframe
    fruits_df = food_data[food_data['Meal_Type'] == 'fruits'].copy()
    # Create drinks dataframe
    drinks_df = food_data[food_data['Meal_Type'] == 'drinks'].copy()
    
    # Filter food choices
    if is_vegan:
        food_data = food_data[food_data['Diet_Restrictions'] == 'vegetarian']
    if no_beef:
        food_data = food_data[food_data['Diet_Restrictions'] != 'beef']
    if low_fat:
        food_data = food_data[food_data['Fat_type'] == 1]
    if is_halal:
        food_data = food_data[food_data['Diet_Restrictions'] != 'pork']
    if high_carb:
        food_data = food_data[food_data['High_carb'] == 1]
    if low_carb:
        food_data = food_data[food_data['High_carb'] == 0]        
    if low_acid:
        food_data = food_data[food_data['Low_acid'] == 1]
    if high_protein:
        food_data = food_data[food_data['Protein_type'] == 3]
    if easily_digestible:
        food_data = food_data[food_data['Easily_Digestable'] == 1]
    if high_fiber:
        food_data = food_data[food_data['High_fiber'] == 1]
        
        
        
    # Create breakfast dataframe
    breakfast_df = food_data[food_data['Meal_Type'] == 'breakfast'].copy()
    # Create mains dataframe
    mains_df = food_data[food_data['Meal_Type'] == 'main'].copy()

    
    breakfast_df.drop(columns=['Meal_Type'], inplace=True)
    mains_df.drop(columns=['Meal_Type'], inplace=True)
    drinks_df.drop(columns=['Meal_Type'], inplace=True)
    fruits_df.drop(columns=['Meal_Type'], inplace=True)
    
    breakfast_data_list = breakfast_df.to_dict(orient="records")
    mains_data_list = mains_df.to_dict(orient="records")
    drinks_data_list = drinks_df.to_dict(orient="records")
    fruits_data_list = fruits_df.to_dict(orient="records")
    
    return breakfast_data_list, mains_data_list, drinks_data_list, fruits_data_list


def calculate_fitness(meal_plan, calorie_goals):
    total_calories = 0
    total_calories_breakfast = 0
    total_calories_lunch = 0
    total_calories_dinner = 0
    calorie_difference = 0
    calorie_diff_BF = 0
    calorie_diff_L = 0
    calorie_diff_D = 0
    total_protein = 0
    total_fat = 0
    total_carbohydrate = 0
    meal_fitness_scores = {}  # Store fitness scores for each meal
    fitness = 0
    mains_BF_ratio = mains_L_ratio = mains_D_ratio = mains_ratio_penalty = 0

    for meal, food_or_drink in meal_plan.items():
        if "breakfast" in meal:
            total_calories_breakfast += food_or_drink["Energy_(kcal)"]
        elif "lunch" in meal:
            total_calories_lunch += food_or_drink["Energy_(kcal)"]
        elif "dinner" in meal:
            total_calories_dinner += food_or_drink["Energy_(kcal)"]

    total_calories = total_calories_breakfast + total_calories_lunch + total_calories_dinner
    # total_protein += food_or_drink["Protein"]
    # total_fat += food_or_drink["Fat"]
    # total_carbohydrate += food_or_drink["Carbohydrate"]

    for meal, goal in calorie_goals.items():
        calorie_diff_BF = round(abs(total_calories_breakfast - calorie_goals["breakfast"]))
        calorie_diff_L = round(abs(total_calories_lunch - calorie_goals["lunch"]))
        calorie_diff_D = round(abs(total_calories_dinner - calorie_goals["dinner"]))
    
    meal_fitness_scores["breakfast"] = calorie_diff_BF
    meal_fitness_scores["lunch"] = calorie_diff_L
    meal_fitness_scores["dinner"] = calorie_diff_D
    
    mains_BF_ratio = meal_plan['breakfast']['Energy_(kcal)'] / total_calories_breakfast
    mains_L_ratio = meal_plan['lunch']['Energy_(kcal)'] / total_calories_lunch
    mains_D_ratio = meal_plan['dinner']['Energy_(kcal)'] / total_calories_dinner
    if mains_BF_ratio < 0.5:
        mains_ratio_penalty += 100
    if mains_L_ratio < 0.5:
        mains_ratio_penalty += 100
    if mains_D_ratio < 0.5:
        mains_ratio_penalty += 100
    if total_calories_breakfast > calorie_goals["breakfast"]:
        mains_ratio_penalty += 500
    if total_calories_lunch > calorie_goals["lunch"]:
        mains_ratio_penalty += 500
    if total_calories_dinner > calorie_goals["dinner"]:
        mains_ratio_penalty += 500
        
    calorie_difference = calorie_diff_BF + calorie_diff_L + calorie_diff_D
    fitness = calorie_difference + mains_ratio_penalty 

    return fitness, meal_fitness_scores


def create_meal_plan(breakfast_data_list, mains_data_list, drinks_data_list, fruits_data_list):
    meal_plan = {}

    # Select a random breakfast item
    breakfast_item = random.choice(breakfast_data_list)
    meal_plan['breakfast'] = breakfast_item

    # Select a random lunch item
    lunch_item = random.choice(mains_data_list)
    meal_plan['lunch'] = lunch_item

    # Select a random dinner item
    dinner_item = random.choice(mains_data_list)
    meal_plan['dinner'] = dinner_item

    # Select a random drink item for each meal
    breakfast_drink_item = random.choice(drinks_data_list)
    lunch_drink_item = random.choice(drinks_data_list)
    dinner_drink_item = random.choice(drinks_data_list)
    meal_plan['breakfast_drink'] = breakfast_drink_item
    meal_plan['lunch_drink'] = lunch_drink_item
    meal_plan['dinner_drink'] = dinner_drink_item

    # Select a random fruit item for each meal
    breakfast_fruit_item = random.choice(fruits_data_list)
    lunch_fruit_item = random.choice(fruits_data_list)
    dinner_fruit_item = random.choice(fruits_data_list)
    meal_plan['breakfast_fruit'] = breakfast_fruit_item
    meal_plan['lunch_fruit'] = lunch_fruit_item
    meal_plan['dinner_fruit'] = dinner_fruit_item
 
    return meal_plan


def run_genetic_algorithm(population_size, num_generations, mutation_rate, breakfast_data_list, mains_data_list, 
                          drinks_data_list, fruits_data_list, calorie_goals):
    population = []
    generation_fitness_Scores = []
    average_fitness_scores = []  # Store the average fitness scores
    best_fitness_scores = []  # Store the best fitness scores

    for _ in range(population_size):
        meal_plan = create_meal_plan(breakfast_data_list, mains_data_list, drinks_data_list, fruits_data_list)
        population.append(meal_plan)

    for generation in range(num_generations):
        population = sorted(population, key=lambda x: calculate_fitness(x, calorie_goals)[0])

        # Select the top-performing meal plans for breeding
        top_performers = population[:20]
        new_population = []

        # Track the best fitness score for this generation
        best_fitness = calculate_fitness(top_performers[0], calorie_goals)[0]
        best_fitness_scores.append(best_fitness)

        # Create new meal plans through crossover and mutation
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(top_performers, 2)
            child = {}
            for meal in calorie_goals.keys():
                crossover_point = random.random()
                if crossover_point < 0.5:
                    child[meal] = parent1[meal]
                    child[meal + "_drink"] = parent1[meal + "_drink"]
                    child[meal + "_fruit"] = parent1[meal + "_fruit"]
                else:
                    child[meal] = parent2[meal]
                    child[meal + "_drink"] = parent2[meal + "_drink"]
                    child[meal + "_fruit"] = parent2[meal + "_fruit"]

                # Apply mutation
                if random.random() < mutation_rate:
                    child[meal] = random.choice(breakfast_data_list) if 'breakfast' in meal else random.choice(mains_data_list)
                    child[meal + "_drink"] = random.choice(drinks_data_list)
                    child[meal + "_fruit"] = random.choice(fruits_data_list)


            new_population.append(child)

        population = new_population

        # Calculate fitness scores for this generation and store them
        fitness_scores = [calculate_fitness(meal_plan, calorie_goals)[0] for meal_plan in population]
        # generation_fitness_scores.append(fitness_scores)

        # Calculate and store the average fitness score for this generation
        average_fitness = sum(fitness_scores) / len(fitness_scores)
        average_fitness_scores.append(average_fitness)

    # Select the best-performing meal plan
    # best_meal_plan = sorted(population, key=lambda x: calculate_fitness(x)[0], reverse=True)[0]
    best_meal_plan = random.choice(top_performers)

    return best_meal_plan, best_fitness_scores, average_fitness_scores


def get_food_filter_args(disease_name):
    disease_rows = disease_filter_data[disease_filter_data['Disease']==disease_name]
    rec:str = disease_rows['Nutrition_Rec_1'].unique()[0]

    if rec in ('-','Low-sodium'):
        return {}, None
    condition = rec.replace('-',' ').replace('_',' ')
    rec = rec.replace('-','_')
    rec = rec.lower()
    # low_fat, low_carb, high_carb, low_acid, high_protein, easily_digestible, high_fiber
    return {rec:True}, condition #foodfilter_kwargs


def recovery_meal_planner(n, userinfo_kwargs, foodfilter_kwargs):
    is_vegan, no_beef, is_halal = userinfo_kwargs['is_vegan'], userinfo_kwargs['no_beef'], userinfo_kwargs['is_halal']
    tdee = BMR(**userinfo_kwargs)
    population_size = 100
    num_generations = 100
    mutation_rate = 0.01
    calorie_goals = {"breakfast": 0.3*tdee, "lunch": 0.4*tdee, "dinner": 0.3*tdee}
    breakfast_data_list, mains_data_list, drinks_data_list, fruits_data_list = filter_data(is_vegan, is_halal, 
                                                                                           no_beef, **foodfilter_kwargs)
    ret = []
    for day in range(1, n + 1):
        best_meal_plan, best_fitness_scores, average_fitness_scores = run_genetic_algorithm(population_size=population_size,
                                                                                            num_generations=num_generations,
                                                                                            mutation_rate=mutation_rate,
                                                                                            breakfast_data_list=breakfast_data_list,
                                                                                            mains_data_list=mains_data_list,
                                                                                            drinks_data_list=drinks_data_list,
                                                                                            fruits_data_list=fruits_data_list,
                                                                                            calorie_goals=calorie_goals)
        day_plan = {'meals':[], 'total':None}
        sum_daily_calories = 0

        # print(f"Day {day} Meal Plan:")
        for meal, item in best_meal_plan.items():
            meal_names = item.get('name', None)
            meal_calories = item.get('Energy_(kcal)', None)
                         
            if meal_names and meal_calories is not None:
                day_plan['meals'].append({
                    'name':meal.replace('_',' ').capitalize(),
                    'items':meal_names,
                    'energy':meal_calories})
                sum_daily_calories += meal_calories

        day_plan['total'] = [round(sum_daily_calories), round(tdee)]

        ret.append(day_plan)
            # if meal_names and meal_calories is not None:
            #     print(f"{meal.capitalize()}:")
            #     print(f"Items: {meal_names}")
            #     print(f"Calories: {meal_calories}")
            #     print()

    return ret
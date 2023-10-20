from json import dumps

def clean_col_name(col_name):
    col_name = col_name.strip()
    col_name = (
        col_name.replace(".1", "")
        .replace("(typhos)", "")
        .replace("yellowish", "yellow")
        .replace("yellowing", "yellow")
    )
    return col_name

def clean_col_name_food(col_name):
    col_name = col_name.strip()
    col_name = (
        col_name.replace(' ','_')
        .replace('-','_')
        .replace('/','_')
    )
    return col_name


def json_res(text:str=None, **kwargs):
    """
    Str or Dict input
    
    Str
    text: raw text message

    others:
    entries: single choise, render as buttons
    options: multiple choise, render as checklist
    table: 
      header: list of column names
      content: list of rows(list)
    suffix: raw text as tail below all other components
    **kwargs: others
    """
    local_dict = locals()
    local_dict.update(local_dict.pop('kwargs'))
    if local_dict['text'] == None:
        del local_dict['text']
    return dumps(local_dict)

def format_mealplan_for_notify(plan:list, day, name):
    plan = plan[day-1]
    ret = f"Hi {name}, this is a reminder of your day {day} meal plan."
    
    meals = plan['meals']
    for meal in meals:
        name = meal['name']
        items = meal['items']
        meal_str = f'\n\n{name}: \n    {items}'
        ret += meal_str
        
    return ret
        
if __name__ == '__main__':
    plan = [{"meals": [{"name": "Breakfast", "items": "Hotcakes with sausage, McDonalds", "energy": 432.96}, {"name": "Breakfast drink", "items": "Orange and mango juice drink", "energy": 117.04}, {"name": "Breakfast fruit", "items": "Orange", "energy": 46.48}, {"name": "Lunch", "items": "Satay, chicken, canned", "energy": 602.19}, {"name": "Lunch drink", "items": "vegemil", "energy": 140.0}, {"name": "Lunch fruit", "items": "Peached Puree, Frozen", "energy": 191.4}, {"name": "Dinner", "items": "Mushroom burger", "energy": 480.84}, {"name": "Dinner drink", "items": "Soya milk powder, instant (HCS)", "energy": 110.4}, {"name": "Dinner fruit", "items": "Mango, honey, raw", "energy": 131.49}], "total": [2253, 2424]}, {"meals": [{"name": "Breakfast", "items": "a.m. Platter, KFC", "energy": 541.2}, {"name": "Breakfast drink", "items": "Tomato juice", "energy": 41.85}, {"name": "Breakfast fruit", "items": "Banana (pisang rajah udang / merah)", "energy": 110.56}, {"name": "Lunch", "items": "Lor mee (NEW)", "energy": 595.2}, {"name": "Lunch drink", "items": "Herbal jelly with sugar syrup", "energy": 70.86}, {"name": "Lunch fruit", "items": "Blueberry filling", "energy": 288.2}, {"name": "Dinner", "items": "Omu rice", "energy": 540.26}, {"name": "Dinner drink", "items": "sweet potato shake", "energy": 175.5}, {"name": "Dinner fruit", "items": "Belimbi", "energy": 1.98}], "total": [2366, 2424]}, {"meals": [{"name": "Breakfast", "items": "biscuit", "energy": 106.8}, {"name": "Breakfast drink", "items": "Chocolate chip frappuccino", "energy": 587.8}, {"name": "Breakfast fruit", "items": "Pomelo, raw, flesh only", "energy": 11.06}, {"name": "Lunch", "items": "Creamy chicken pasta", "energy": 765.4}, {"name": "Lunch drink", "items": "Root beer", "energy": 135.66}, {"name": "Lunch fruit", "items": "Orange, valencia, raw", "energy": 61.47}, {"name": "Dinner", "items": "Sweet and sour prawn", "energy": 437.08}, {"name": "Dinner drink", "items": "Low fat chocolate milk", "energy": 204.82}, {"name": "Dinner fruit", "items": "Pear, canned in syrup", "energy": 63.18}], "total": [2373, 2424]}, {"meals": [{"name": "Breakfast", "items": "Kaya waffle", "energy": 417.48}, {"name": "Breakfast drink", "items": "Buttermilk", "energy": 149.36}, {"name": "Breakfast fruit", "items": "Papaya, exotica, Hawaii", "energy": 148.77}, {"name": "Lunch", "items": "Beef Lasagne, Pizzahut", "energy": 917.0}, {"name": "Lunch drink", "items": "orange juice (sugar-free)", "energy": 39.0}, {"name": "Lunch fruit", "items": "Kundur, candied", "energy": 11.52}, {"name": "Dinner", "items": "Prawn tempura udon soup", "energy": 573.5}, {"name": "Dinner drink", "items": "Milk powder, skimmed/low fat, high calcium, instant", "energy": 19.53}, {"name": "Dinner fruit", "items": "Apple, stewed with sugar", "energy": 127.83}], "total": [2404, 2424]}, {"meals": [{"name": "Breakfast", "items": "Fried bee hoon with chicken and egg", "energy": 369.2}, {"name": "Breakfast drink", "items": "Isotonic drink (HCS)", "energy": 82.5}, {"name": "Breakfast fruit", "items": "Prune, dried, cooked, without sugar", "energy": 226.84}, {"name": "Lunch", "items": "Pad thai", "energy": 844.36}, {"name": "Lunch drink", "items": "lemon juice", "energy": 38.0}, {"name": "Lunch fruit", "items": "pear (100g)", "energy": 46.0}, {"name": "Dinner", "items": "Rice dumpling with meat filling", "energy": 386.65}, {"name": "Dinner drink", "items": "Chocolate milk bubble tea", "energy": 319.44}, {"name": "Dinner fruit", "items": "Date, Chinese, black, dried", "energy": 4.96}], "total": [2318, 2424]}]
    print( format_mealplan_for_notify(plan, 1,' usera') )
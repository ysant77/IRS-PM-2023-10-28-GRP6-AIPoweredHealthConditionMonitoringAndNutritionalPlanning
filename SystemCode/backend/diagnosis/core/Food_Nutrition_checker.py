from django.apps import apps

food_data = apps.get_app_config("diagnosis").food_df_for_checker

def search_food_data(keyword):
    keyword = keyword.lower()
    matching_items = food_data[food_data['name'].str.lower().str.contains(keyword, na=False)]

    if matching_items.empty:
        # print(f"No matching items found for '{keyword}'. Please try again.")
        return None, False
    else:
        # print("Matching items found:")
        # display_results(matching_items)
        return matching_items, True

def display_results(matching_items):
    print("Results:")
    for i, item in enumerate(matching_items['name'], start=1):
        print(f"{i}. {item}")
    print()

def food_nutrition_checker(matching_items, confirmation):
    # keyword = input("Enter a keyword to search for a food item (type 'exit' to quit): ")

    # if keyword.lower() == 'exit':
    #     break

    # confirmation = input("Enter the number corresponding to the item you want to check, or type 'no' to search again: ")
    if confirmation.lower() == -1:
            return False
    selected_item = matching_items.iloc[int(confirmation) - 1]
    # print(f"\nYou've selected: {selected_item['name']}\n")

    # Print nutrition values
    # nutrition_values = selected_item.drop('name')  # Exclude the 'name' column
    # print("Nutrition values:")
    # print(nutrition_values)

    # Return a dictionary with relevant information
    return {
        'selected_item': selected_item['name'],
        'nutrition_values': selected_item.drop('name').to_dict(),
    }


if __name__ == "__main__":
    food_nutrition_checker()
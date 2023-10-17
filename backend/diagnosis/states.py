# states.py
from channels.db import database_sync_to_async
from django.forms.models import model_to_dict

from .core.nlp_chatbot import extract_symptoms, get_relevant_symptoms, perform_diagnosis
from .core.single_meal_planner import get_calorie_goal
from .core.Food_Nutrition_checker import search_food_data
from .core.GA_Meal_Planner_r2_5 import recovery_meal_planner, get_food_filter_args

# from .consumers import ChatConsumer
from .databaseutil import get_user

DEFALUT_MSG = 'Sorry, please try again.'

class BaseState:
    no_preprocess = False

    def __init__(self, session, user_message, username, consumer=None):
        if not self.no_preprocess:
            user_message = str(user_message)
            user_message = user_message.lower()
            user_message = user_message.strip().strip('.')

        self.session = session
        self.user_message = user_message
        self.username = username

        ## =========NOTICE: Every `self.consumer.xxx` code is subject to change. =====================
        ## Using this attr (taking the Consumer in and save temp data in it) 
        ##      is a temperary solution to avoid complexity.
        ## What data should be stored in database is to be discussed.
        ## For now, these temporary data will be lost once the connection closed,
        ##      and users won't be able to continue the dialog even when reconnect the websocket.
        ##============================================================================================
        self.consumer = consumer
    
    @database_sync_to_async
    def _save_session(self):
        self.session.save()

    def respond(self):
        raise NotImplementedError()


class GreetingState(BaseState):
    async def respond(self):
        if self.user_message == 'meal planning':
            self.session.conversation_state = 'AskMealState'
            await self._save_session()
            return {'text':'Please select which meal to plan.',
                    'entries':['Breakfast','Lunch','Dinner']}
        
        if self.user_message == 'symptom diagnosis':
            self.session.conversation_state = 'AskSymptomsState'
            await self._save_session()

            return 'Please describe symptoms you are facing.'
        return DEFALUT_MSG


class AskSymptomsState(BaseState):
    async def respond(self):
        # Here you should use your model or function to identify potential matching symptoms
        extracted_symptoms = extract_symptoms(self.user_message)
        if not extracted_symptoms:
            return "Sorry, I couldn't recognize your symptoms. Try again."
        relevant_symptoms = get_relevant_symptoms(extracted_symptoms)

        if relevant_symptoms:
            self.session.conversation_state = 'ConfirmSymptomsState'
            self.session.extracted_symptoms = ",".join(relevant_symptoms)
            await self._save_session()
            return {'text':"Are you facing any of these symptoms?",
                    'options':relevant_symptoms}
        else:
            return await self._diagnose(extracted_symptoms)

    async def _diagnose(self, symptoms):
        disease_list = perform_diagnosis(symptoms)

        if len(disease_list) == 0:
            self.session.conversation_state = 'AskSymptomsState'
            await self._save_session()
            return 'There seems to be no disease diagnosed. You may try to describe your symptoms again.'
        
        self.consumer.top_disease = disease_list[0][0]

        msg = {'table':{'header':'diagnosis', 'content':disease_list},
                'suffix':'diagnosis'}
        msg2 = {'text':'Would you like to checkout the meal plan?',
                'confirm':['Yes','No']}
        
        self.session.conversation_state = 'DiagnoseState'
        await self._save_session()
        return {'msg':msg, 'next':msg2}
           

class ConfirmSymptomsState(AskSymptomsState):
    async def respond(self):
        combined_symptoms = self.session.extracted_symptoms.split(",")
        if "continue" not in self.user_message:
            confirmed_symptoms = self.user_message.split(",")
            combined_symptoms += confirmed_symptoms
            combined_symptoms = list(set(combined_symptoms))

        return await self._diagnose(combined_symptoms)


class DiagnoseState(BaseState):
    async def respond(self):       
        if self.user_message == "yes":
            self.consumer.plan_days_no = 3

            user = await get_user(self.consumer.user.uid)
            user_dict = model_to_dict(user)

            filter_dict, condition = get_food_filter_args(self.consumer.top_disease)

            meal_plan_output = recovery_meal_planner(n=self.consumer.plan_days_no,
                                                     userinfo_kwargs=user_dict,
                                                     foodfilter_kwargs=filter_dict)

            self.session.conversation_state = 'GreetingState'
            await self._save_session()

            message = f' you are recommended to have {condition.lower()} food. ' if filter_dict else ' there is no limitation of your food choise. '
            return {'text':f'Here is your {self.consumer.plan_days_no}-day meal plan.',
                    'plan':meal_plan_output,
                    'suffix':f'This meal plan was generated considering your possible disease {self.consumer.top_disease},'
                            +message+'This meal plan is just for reference.'}
        
        if self.user_message == "no":
            self.session.conversation_state = 'GreetingState'
            await self._save_session()
            return {'text':"Is there anything else you'd like to do?",
                    'entries':['Meal planning','Symptom diagnosis']}


# class MealPlannerState(BaseState):
#     async def respond(self):
#         self.session.conversation_state = 'GreetingState'
#         await self._save_session()
#         return "Is there anything else you'd like to know or ask?"
    

class AskMealState(BaseState):
    async def respond(self):
        if self.user_message in ['breakfast', 'lunch', 'dinner']:
            user = await get_user(self.consumer.user.uid)
            user_dict = model_to_dict(user)
            self.consumer.remain_calorie = get_calorie_goal(user_meal_type=self.user_message, **user_dict)

            self.consumer.meal = self.user_message
            self.consumer.meal_plan = []

            self.session.conversation_state = 'QueryFoodState'
            await self._save_session()
            return f'According your physical information and your preferences, you need {self.consumer.remain_calorie} kcal of energy consumption.\nEnter a keyword to search for a food item.'
        return DEFALUT_MSG
    

class QueryFoodState(BaseState):
    async def respond(self):
        matching_items, not_none = search_food_data(self.user_message)
        if not not_none:
            return f'No matching items for {self.user_message}. Try again.'
        
        self.consumer.food_query = matching_items
        query_list = [item for item in matching_items['name']]

        self.session.conversation_state = 'AddFoodState'
        await self._save_session()
        return {'text':'Select one from food query list.',
                'dropdown':query_list}
    

class AddFoodState(BaseState):
    no_preprocess = True

    async def respond(self):
        food_query = self.consumer.food_query
        selected_food = food_query[food_query['name']==self.user_message]
        self.consumer.meal_plan.append(selected_food)
        energy = selected_food['Energy_(kcal)'].iloc[0]

        self.consumer.remain_calorie -= round(energy)
        remain_cal = self.consumer.remain_calorie

        if remain_cal > 0:
            txt = f"You have {remain_cal} kcal of calories remaining for your {self.consumer.meal} meal plan."
        else:
            txt = f"You have exceeded your calorie goals for your {self.consumer.meal} meal plan by {-remain_cal}."
        prompt = f" Continue adding food item?"

        self.session.conversation_state = 'AfterQueryState'
        await self._save_session()
        return {'text':f"You've selected:",
                'table':{'header':'none','content':[[self.user_message, 'with', energy, 'kcal']]},
                'suffix':txt+prompt,
                'confirm':['Yes','No']}


class AfterQueryState(BaseState):
    async def respond(self):
        if self.user_message == 'yes':
            self.session.conversation_state = 'QueryFoodState'
            await self._save_session()
            return 'Enter a keyword to search for a food item.'
        
        if self.user_message == 'no':
            self.session.conversation_state = 'GreetingState'
            await self._save_session()
            # show meal plan
            meal_plan = [[item['name'].iloc[0], item['Energy_(kcal)'].iloc[0]] for item in self.consumer.meal_plan]
            msg1 = {'text':f'Here is your {self.consumer.meal} meal plan.',
                    'table':{'header':'mealPlan','content':meal_plan}}
            msg2 = {'text':'What else would you like to do?',
                    'entries':['Meal planning','Symptom diagnosis']}
            return {'msg':msg1, 'next':msg2}
        return DEFALUT_MSG
        
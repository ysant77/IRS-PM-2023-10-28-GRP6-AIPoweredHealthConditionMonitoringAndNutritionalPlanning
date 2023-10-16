from channels.db import database_sync_to_async
from .utils import format_diagnosis_for_display
from .nlp_chatbot import extract_symptoms, get_relevant_symptoms, perform_diagnosis


class BaseState:
    def __init__(self, session, user_message, username):
        self.session = session
        self.user_message = user_message
        self.username = username
    
    @database_sync_to_async
    def _save_session(self):
        self.session.save()

    def respond(self):
        raise NotImplementedError()

class GreetingState(BaseState):

    async def respond(self):
        self.session.username = self.username
        self.session.conversation_state = 'AskSymptomsState'

        await self._save_session()#self.session.save()
        return f"Hello, {self.username}! What symptoms are you facing?"

class AskSymptomsState(BaseState):

    async def respond(self):
        
        self.user_message = str(self.user_message).lower()
        extracted_symptoms = extract_symptoms(self.user_message)
        if not extracted_symptoms:
            return "Sorry, I couldn't recognize your symptoms. Try again."
        relevant_symptoms = get_relevant_symptoms(extracted_symptoms)

        if relevant_symptoms:
            self.session.conversation_state = 'ConfirmSymptomsState'
            self.session.extracted_symptoms = ",".join(relevant_symptoms)
            await self._save_session()
            return {"type": "multi_select", "bot_message": "Are you facing any of these symptoms? or type 'continue' for diagnosis",
                     "options": relevant_symptoms}
        else:
            self.session.conversation_state = 'DiagnoseState'
            await self._save_session()
            return await self._diagnose(extracted_symptoms)

    async def _diagnose(self, symptoms):
        disease_list = perform_diagnosis(symptoms)
        return format_diagnosis_for_display(disease_list)
        

    

class ConfirmSymptomsState(BaseState):

    async def respond(self):
        combined_symptoms = self.session.extracted_symptoms.split(",")
        self.user_message = str(self.user_message).lower()
        if "continue" not in self.user_message:
            confirmed_symptoms = self.user_message.split(",")
            combined_symptoms += confirmed_symptoms
            combined_symptoms = list(set(combined_symptoms))

        self.session.conversation_state = 'DiagnoseState'
        await self._save_session()
        return await self._diagnose(combined_symptoms)


    async def _diagnose(self, symptoms):
        disease_list = perform_diagnosis(symptoms)
        return format_diagnosis_for_display(disease_list)

class DiagnoseState(BaseState):
    async def respond(self):
        self.user_message = str(self.user_message).lower()
        
        if self.user_message == "yes":
            self.session.conversation_state = 'MealPlannerState'
            await self._save_session()
            return "Here's a meal plan for you. \nType anything to continue."
        elif self.user_message == "no":
            print('inside the no part ')
            self.session.conversation_state = 'GreetingState'
            await self._save_session()
            return "Is there anything else you'd like to know or ask?"
        else:
            self.session.conversation_state = 'GreetingState'
            await self._save_session()
            return "I am sorry, I didn't understand that. Is there anything else you'd like to know or ask?"

class MealPlannerState(BaseState):
    async def respond(self):
        self.session.conversation_state = 'GreetingState'
        await self._save_session()
        return "Is there anything else you'd like to know or ask?"



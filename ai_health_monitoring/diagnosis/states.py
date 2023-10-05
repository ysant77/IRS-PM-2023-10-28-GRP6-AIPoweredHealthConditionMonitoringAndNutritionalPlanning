# states.py
from channels.db import database_sync_to_async
from .utils import format_diagnosis_for_display
from .nlp_chatbot import extract_symptoms, get_relevant_symptoms, perform_diagnosis


class BaseState:
    def __init__(self, session, user_message):
        self.session = session
        self.user_message = user_message
    
    @database_sync_to_async
    def _save_session(self):
        self.session.save()

    def respond(self):
        raise NotImplementedError()

class GreetingState(BaseState):

    async def respond(self):
        self.session.username = self.user_message
        self.session.conversation_state = 'AskSymptomsState'
        await self._save_session()#self.session.save()
        return f"Hello, {self.user_message}! What symptoms are you facing?"

class AskSymptomsState(BaseState):

    async def respond(self):
        # Here you should use your model or function to identify potential matching symptoms
        self.user_message = str(self.user_message).lower()
        extracted_symptoms = extract_symptoms(self.user_message)
        if not extracted_symptoms:
            return "Sorry, I couldn't recognize your symptoms. Try again."
        relevant_symptoms = get_relevant_symptoms(extracted_symptoms)

        if relevant_symptoms:
            self.session.conversation_state = 'ConfirmSymptomsState'
            self.session.extracted_symptoms = ",".join(relevant_symptoms)
            await self._save_session()#self.session.save()
            additional_symptoms = ", ".join(relevant_symptoms)
            return f"Are you also facing {additional_symptoms}? Or type 'continue' for diagnosis."
        else:
            self.session.conversation_state = 'DiagnoseState'
            await self._save_session()#self.session.save()
            return await self._diagnose(extracted_symptoms)

    async def _diagnose(self, symptoms):
        disease_list = perform_diagnosis(symptoms)
        return format_diagnosis_for_display(disease_list)
        

    

class ConfirmSymptomsState(BaseState):

    async def respond(self):
        combined_symptoms = self.session.extracted_symptoms.split(",")
        self.user_message = str(self.user_message).lower()
        if "continue" not in self.user_message:
            confirmed_symptoms = self.user_message.split(", ")
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
        self.session.conversation_state = 'GreetingState'
        await self._save_session()#self.session.save()
        return "Is there anything else you'd like to know or ask?"

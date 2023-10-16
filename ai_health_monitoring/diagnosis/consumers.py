import django
django.setup()
from channels.db import database_sync_to_async


import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .models import ChatSession, ChatMessage
from .states import GreetingState, AskSymptomsState, ConfirmSymptomsState, DiagnoseState, MealPlannerState

STATE_MAPPING = {
    "GreetingState": GreetingState,
    "AskSymptomsState": AskSymptomsState,
    "ConfirmSymptomsState": ConfirmSymptomsState,
    "DiagnoseState": DiagnoseState,
    "MealPlannerState": MealPlannerState
}

@database_sync_to_async
def get_or_create_session(session_id, username):
    return ChatSession.objects.get_or_create(id=session_id, username=username)

@database_sync_to_async
def create_initial_chat_message(session, is_user, content):
    return ChatMessage.objects.create(session=session, is_user=is_user, content=content)



class ChatConsumer(AsyncWebsocketConsumer):
    
    async def connect(self):
        if not self.scope["user"].is_authenticated:
            
            await self.close()
            return
        self.username = self.scope['user'].username
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.session, created = await get_or_create_session(self.session_id, self.username)

        await self.accept()

        if created:
            initial_msg = f"Hello {self.username}, What symptoms are you facing?"
            chat_message = await create_initial_chat_message(self.session, False, initial_msg)
            await self.send(text_data=json.dumps({
                'bot_message': initial_msg
            }))

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        user_message = text_data_json['message']

        if user_message.lower() == "exit":
            await self.close()
            return

        response_data = {}
        state_cls = STATE_MAPPING.get(self.session.conversation_state)
        if state_cls:
            state_instance = state_cls(self.session, user_message, self.username)
            bot_response = await state_instance.respond()
            
            if isinstance(bot_response, dict) and bot_response.get("type") == "multi_select":
                response_data = bot_response
            else:
                response_data = {
                    'type': 'text',
                    'bot_message': bot_response
                }
            if isinstance(bot_response, dict):
                bot_response_content = bot_response.get('bot_message', '')
            else:
                bot_response_content = bot_response
            chat_message = await create_initial_chat_message(self.session, True, user_message)
            chat_message = await create_initial_chat_message(self.session, False, bot_response_content)
            
        else:
            response_data = {
                'bot_message': "I'm sorry, I didn't understand that."
            }
            
        await self.send(text_data=json.dumps(response_data))

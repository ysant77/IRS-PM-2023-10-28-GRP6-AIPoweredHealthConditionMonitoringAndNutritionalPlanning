# consumers.py
import django
django.setup()
#import os
#os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
from channels.db import database_sync_to_async


import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .models import ChatSession, ChatMessage
from .states import GreetingState, AskSymptomsState, ConfirmSymptomsState, DiagnoseState

STATE_MAPPING = {
    "GreetingState": GreetingState,
    "AskSymptomsState": AskSymptomsState,
    "ConfirmSymptomsState": ConfirmSymptomsState,
    "DiagnoseState": DiagnoseState
}

@database_sync_to_async
def get_or_create_session(session_id):
    return ChatSession.objects.get_or_create(id=session_id)

@database_sync_to_async
def create_initial_chat_message(session, is_user, content):
    return ChatMessage.objects.create(session=session, is_user=is_user, content=content)



class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.session, created = await get_or_create_session(self.session_id) #ChatSession.objects.get_or_create(id=self.session_id)

        await self.accept()

        if created:
            initial_msg = "Hello! Please enter your username."
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

        state_cls = STATE_MAPPING.get(self.session.conversation_state)
        if state_cls:
            state_instance = state_cls(self.session, user_message)
            bot_response =  await state_instance.respond()
        else:
            bot_response = "I'm sorry, I didn't understand that."
        
        print(bot_response)
        chat_message = await create_initial_chat_message(self.session, True, user_message)
        chat_message = await create_initial_chat_message(self.session, False, bot_response)

        await self.send(text_data=json.dumps({
            'bot_message': bot_response
        }))

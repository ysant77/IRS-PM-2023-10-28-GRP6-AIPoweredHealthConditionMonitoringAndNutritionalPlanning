# consumers.py
import django
django.setup()
#import os
#os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
from channels.db import database_sync_to_async


import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .models import ChatSession, ChatMessage
from .states import GreetingState, AskSymptomsState, ConfirmSymptomsState, DiagnoseState, PostDiagnoseState

STATE_MAPPING = {
    "GreetingState": GreetingState,
    "AskSymptomsState": AskSymptomsState,
    "ConfirmSymptomsState": ConfirmSymptomsState,
    "DiagnoseState": DiagnoseState,
    "PostDiagnoseState": PostDiagnoseState
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
    # async def receive(self, text_data):
    #     text_data_json = json.loads(text_data)
    #     user_message = text_data_json['message']

    #     if user_message.lower() == "exit":
    #         await self.close()
    #         return

    #     messages_to_send = []

    #     previous_state = self.session.conversation_state
    #     state_cls = STATE_MAPPING.get(previous_state)
    #     if state_cls:
    #         state_instance = state_cls(self.session, user_message)
    #         bot_response = await state_instance.respond()

    #         # Append the bot's initial response to the messages_to_send list
    #         if isinstance(bot_response, dict):
    #             messages_to_send.append(bot_response)
    #         else:
    #             messages_to_send.append({'type': 'text', 'bot_message': bot_response})

    #         # If the state has changed to DiagnoseState, add the additional message.
    #         if previous_state != self.session.conversation_state and self.session.conversation_state == "DiagnoseState":
    #             follow_up_msg = "Is there anything else you'd like to know or ask?"
    #             messages_to_send.append({'type': 'text', 'bot_message': follow_up_msg})
                
    #         if isinstance(bot_response, dict):
    #             bot_message_content = bot_response.get('bot_message', '')
    #         else:
    #             bot_message_content = bot_response

    #         chat_message = await create_initial_chat_message(self.session, True, user_message)
    #         chat_message = await create_initial_chat_message(self.session, False, bot_message_content)


    #     else:
    #         messages_to_send.append({
    #             'bot_message': "I'm sorry, I didn't understand that."
    #         })

    #     # Combine and send all messages at the end
    #     for message in messages_to_send:
    #         await self.send(text_data=json.dumps(message))
        
    #     print(self.session.conversation_state)

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        user_message = text_data_json['message']

        if user_message.lower() == "exit":
            await self.close()
            return

        response_data = {}
        # Check if a diagnosis was just given

        print('BEFORE STATE MAPPING ',self.session.conversation_state)
        state_cls = STATE_MAPPING.get(self.session.conversation_state)
        if state_cls:
            state_instance = state_cls(self.session, user_message)
            bot_response = await state_instance.respond()
            print(bot_response)
            print(self.session.conversation_state)
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

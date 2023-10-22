# consumers.py
import django
django.setup()
#import os
#os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

import json
from channels.generic.websocket import AsyncWebsocketConsumer
from django.forms.models import model_to_dict

from .utils import json_res
from .states import (GreetingState,AskSymptomsState,ConfirmSymptomsState,DiagnoseState,
                     AskMealState,QueryFoodState,AddFoodState,AfterQueryState)
from .states import BaseState
from .databaseutil import * 

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = await get_user(uid=self.scope['user'].id)
        self.user_notify, _ = await get_or_create_user_notify(user=self.user)
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.session, created = await get_or_create_session(self.session_id, self.user.uid) #ChatSession.objects.get_or_create(id=self.session_id)
        
        await self.accept()
        
        if created:
            bot_response = json_res(text=f"Hello {self.user.name}! What do you want to do?",
                                            entries=['Meal planning','Symptom diagnosis'])
            await create_chat_message(self.session, False, bot_response)
            await self.send(bot_response)

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        user_message = text_data_json['message']

        # Check if a diagnosis was just given
        state_cls: BaseState = eval(self.session.conversation_state)
        if issubclass(state_cls, BaseState):
            state_instance = state_cls(self.session, user_message, self.user, self.user_notify, self)
            bot_response = await state_instance.respond()
            if isinstance(bot_response, str):
                bot_response = json_res(bot_response)
            else:
                bot_response = json_res(**bot_response)

            await create_chat_message(self.session, True, user_message)
            await create_chat_message(self.session, False, bot_response)
            
        else:
            bot_response = json_res("I'm sorry, I didn't understand that.")
            
        await self.send(bot_response)


# For certain reasons, http request API cannot get the logged-in user info properly.
# So used a ws instead, a temperary solution.
class CheckLoginConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

        if not self.scope['user'].is_authenticated:
            # Redirect to login page
            await self.send(json_res(**{'redirect':'login'}))
            await self.close()
            return
        
        uid = self.scope['user'].id
        user, new = await get_or_create_user(uid)
        if new:
            username = self.scope['user'].username
            await set_user_name(uid=uid, name=username)
            
        user_dict = model_to_dict(user, exclude=['name','uid'])
        if not all([user_dict[k] for k in ('gender','age','height','weight')]):
            # Redirect to info page
            await self.send(json_res(**{'redirect':'info'}))
            await self.close()
            return
        
        await self.send(json_res(**{'status':True}))
        await self.close()
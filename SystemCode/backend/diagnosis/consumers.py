# consumers.py
import django
django.setup()
#import os
#os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

import json
from channels.generic.websocket import AsyncWebsocketConsumer

from .utils import json_res
from .states import *
from .databaseutil import * 

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

        # if not self.scope["user"].is_authenticated:
        #     # Redirect to login page
        #     await self.send(json_res('Not login'))
        #     await self.close()
        #     return
        # else:
        #     await self.send(json_res('login'))
        #     await self.close()
        #     return
        
        # self.user, _ = await get_or_create_user(self.scope['user'].id)
        # if not self.user.name:
        #     await update_user(**{'uid':self.user.uid,
        #                          'name':self.scope['user'].username})
        #     await self.send(JSONresponse(**{'redirect':'info'}))
            
        # else:
        self.user = await get_user(uid=self.scope['user'].id)
        self.username = self.user.name
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.session, created = await get_or_create_session(self.session_id, self.user.uid) #ChatSession.objects.get_or_create(id=self.session_id)
        if created:
            await self.send(json_res(text=f"Hello {self.username}! What do you want to do?",
                                            entries=['Meal planning','Symptom diagnosis']))

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        user_message = text_data_json['message']

        # if user_message.lower() == "exit":
        #     await self.close()
        #     return

        # Check if a diagnosis was just given
        #print('BEFORE STATE MAPPING ',self.session.conversation_state)
        # state_cls = STATE_MAPPING.get(self.session.conversation_state)
        state_cls = eval(self.session.conversation_state)
        if state_cls:
            state_instance = state_cls(self.session, user_message, self.username, self)
            bot_response = await state_instance.respond()
            if isinstance(bot_response, str):
                bot_response = json_res(bot_response)
            else:
                bot_response = json_res(**bot_response)
            #print(bot_response)
            #print(self.session.conversation_state)
            chat_message = await create_chat_message(self.session, True, user_message)
            chat_message = await create_chat_message(self.session, False, bot_response)
            
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
        user, _ = await get_or_create_user(uid)
        user_dict = model_to_dict(user)
        if not all([user_dict[k] for k in ('name','gender','age','height','weight')]):
            # Redirect to info page
            await self.send(json_res(**{'redirect':'info'}))
            await self.close()
            return
        
        await self.send(json_res(**{'status':True}))
        await self.close()
        return
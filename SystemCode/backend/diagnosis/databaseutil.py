from channels.db import database_sync_to_async

from .models import *

#this decorator allows us to call the synchronous django db operation in the async methods
@database_sync_to_async
def get_or_create_session(session_id, uid):
    return ChatSession.objects.get_or_create(id=session_id, uid=uid)

@database_sync_to_async
def create_chat_message(session, is_user, content):
    return ChatMessage.objects.create(session=session, is_user=is_user, content=content)

@database_sync_to_async
def get_or_create_user(uid, **kwargs):
    return Users.objects.get_or_create(uid=uid, **kwargs)

@database_sync_to_async
def get_user(uid):
    return Users.objects.get(uid=uid)

@database_sync_to_async
def set_user_name(uid, name):
    users = Users.objects.filter(uid=uid)
    users.update(name=name)

@database_sync_to_async
def get_or_create_user_notify(user):
    return UserNotify.objects.get_or_create(user=user)
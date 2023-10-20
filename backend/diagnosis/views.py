from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.forms.models import model_to_dict

import asyncio
from json import loads

from .models import *
from .databaseutil import *
from .telegram import filter_msg, send_msg
from .utils import format_mealplan_for_notify

def login_view(request):
    if request.user.is_authenticated:
        return redirect('chat_page')
    return render(request, 'diagnosis/google-login.html')


@login_required
def chat_page(request):
    return redirect('http://localhost:3000/')


def curr_user_info(request, *args):
    uid = request.scope['user'].id
    user = Users.objects.get(uid=uid)
    return JsonResponse(model_to_dict(user))


def update_user_info(request, *args):
    data = request.POST.get('data', None)
    status = False
    if data is not None:
        status = True
        data = loads(data)
        uid = data.pop('uid')
        user = Users.objects.filter(uid=uid)
        user.update(**data)

    return JsonResponse({'status':status})

def curr_user_notify(request):
    uid = request.scope['user'].id
    user = Users.objects.get(uid=uid)
    usernotify = UserNotify.objects.get(user=user)
    has_chat_id = True if usernotify.chat_id else False
    return JsonResponse({'username':user.name, 'configured':has_chat_id, 'notify_on':usernotify.notify_on})

def curr_user_verify(request):
    code = request.POST.get('code', None)
    if code is None:
        return JsonResponse({'failed':True})
    
    # Search if Telegram bot has received the code msg
    found, chat_id = asyncio.run(filter_msg(code))
    if not found:
        return JsonResponse({'verified':False})
    uid = request.scope['user'].id
    usernotify = UserNotify.objects.filter(user__uid=uid)
    usernotify.update(chat_id=chat_id, notify_on=True)
    return JsonResponse({'verified':True})

def send_tele_meal_plan(request):
    msg = request.POST.get('msg', None)
    if msg != 'msg':
        return JsonResponse({'failed':True})
    
    uid = request.scope['user'].id
    user = Users.objects.get(uid=uid)
    usernotify = UserNotify.objects.get(user=user)
    username = user.name
    chat_id = usernotify.chat_id
    latest_meal_plan = usernotify.latest_meal_plan
    
    msg = format_mealplan_for_notify(latest_meal_plan['plan'],1,username)
    asyncio.run( send_msg(chat_id, msg) )
    return JsonResponse({'sent':True})
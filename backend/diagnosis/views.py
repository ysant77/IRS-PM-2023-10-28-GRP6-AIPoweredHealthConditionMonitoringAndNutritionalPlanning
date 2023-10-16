from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.forms.models import model_to_dict

from json import loads

from .models import *
from .databaseutil import *

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
    

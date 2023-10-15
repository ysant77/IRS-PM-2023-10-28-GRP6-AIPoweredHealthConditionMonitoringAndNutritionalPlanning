from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

from .models import *

def login_view(request):
    if request.user.is_authenticated:
        return redirect('chat_page')
    return render(request, 'diagnosis/google-login.html')

@login_required
def chat_page(request):
    return redirect('http://localhost:3000/')

@login_required
def curr_user_info(request):
    uid = request.POST['user'].id
    user = Users.objects.get(uid=uid)
    return JsonResponse('retstr')

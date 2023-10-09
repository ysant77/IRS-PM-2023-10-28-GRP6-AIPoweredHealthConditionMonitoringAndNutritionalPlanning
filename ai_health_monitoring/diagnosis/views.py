from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

def login_view(request):
    if request.user.is_authenticated:
        return redirect('chat_page')
    return render(request, 'diagnosis/google-login.html')

@login_required
def chat_page(request):
    return render(request, 'diagnosis/chat.html')

"""
URL configuration for ai_health_monitoring project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include
from diagnosis import views, consumers
from django.contrib import admin

urlpatterns = [
    path('', views.login_view, name='login_view'),  # Login page
    path('chat/', views.chat_page, name='chat_page'),  # Chat page after successful login
    path('accounts/', include('allauth.urls')),   # For django-allauth

    path('admin/',admin.site.urls),

    path('api/currentuser/', consumers.UserInfoConsumer.as_asgi()),
]

# models.py

from django.db import models

class ChatSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    uid = models.PositiveBigIntegerField(null=True)
    extracted_symptoms = models.TextField(blank=True, null=True)
    conversation_state = models.CharField(max_length=100, default='GreetingState')

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    is_user = models.BooleanField()
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

class Users(models.Model):
    uid = models.PositiveBigIntegerField(primary_key=True)
    name = models.CharField(max_length=100, null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    age = models.IntegerField(null=True)
    height = models.CharField(max_length=10, null=True, blank=True)
    weight = models.CharField(max_length=10, null=True, blank=True)
    exec_lvl = models.IntegerField(default=3, verbose_name='exercise level')
    weight_goal = models.IntegerField(default=2)
    is_vegan = models.BooleanField(default=False)
    no_beef = models.BooleanField(default=False)
    is_halal = models.BooleanField(default=False)
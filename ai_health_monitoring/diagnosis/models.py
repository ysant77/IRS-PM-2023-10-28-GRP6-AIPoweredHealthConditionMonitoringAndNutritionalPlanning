# models.py

from django.db import models

class ChatSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    username = models.CharField(max_length=100, blank=True, null=True)
    extracted_symptoms = models.TextField(blank=True, null=True)
    conversation_state = models.CharField(max_length=100, default='AskSymptomsState')

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    is_user = models.BooleanField()
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

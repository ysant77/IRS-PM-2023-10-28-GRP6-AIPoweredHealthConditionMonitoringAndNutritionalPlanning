# Generated by Django 4.2.6 on 2023-10-05 09:52

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("diagnosis", "0002_alter_chatsession_conversation_state"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatsession",
            name="extracted_symptoms",
            field=models.TextField(blank=True, null=True),
        ),
    ]

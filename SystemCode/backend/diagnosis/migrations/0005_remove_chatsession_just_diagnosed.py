# Generated by Django 4.2.6 on 2023-10-05 15:51

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("diagnosis", "0004_chatsession_just_diagnosed"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="chatsession",
            name="just_diagnosed",
        ),
    ]

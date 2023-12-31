# Generated by Django 4.2.6 on 2023-10-14 07:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('diagnosis', '0006_alter_chatsession_conversation_state'),
    ]

    operations = [
        migrations.CreateModel(
            name='Users',
            fields=[
                ('uid', models.PositiveBigIntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(blank=True, max_length=100, null=True)),
                ('gender', models.CharField(blank=True, max_length=10, null=True)),
                ('age', models.IntegerField(null=True)),
                ('height', models.CharField(blank=True, max_length=10, null=True)),
                ('weight', models.CharField(blank=True, max_length=10, null=True)),
                ('exec_lvl', models.IntegerField(default=3, verbose_name='exercise level')),
                ('weight_goal', models.IntegerField(default=2)),
                ('is_vegan', models.BooleanField(default=False)),
                ('no_beef', models.BooleanField(default=False)),
                ('is_halal', models.BooleanField(default=False)),
            ],
        ),
        migrations.RemoveField(
            model_name='chatsession',
            name='username',
        ),
        migrations.AddField(
            model_name='chatsession',
            name='uid',
            field=models.PositiveBigIntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='chatsession',
            name='conversation_state',
            field=models.CharField(default='GreetingState', max_length=100),
        ),
    ]

# Generated by Django 4.1.7 on 2024-08-13 12:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("court_detection", "0002_basketballgame_corners_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="basketballgame",
            name="n",
            field=models.IntegerField(default=0),
        ),
    ]

# Generated by Django 4.0.6 on 2022-07-17 17:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classattendence', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='student',
            name='roll_no',
            field=models.IntegerField(default=-1),
        ),
    ]

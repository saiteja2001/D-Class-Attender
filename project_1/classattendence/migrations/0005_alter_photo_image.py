# Generated by Django 4.0.6 on 2022-07-19 13:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classattendence', '0004_alter_photo_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='image',
            field=models.FileField(upload_to='dataset/'),
        ),
    ]
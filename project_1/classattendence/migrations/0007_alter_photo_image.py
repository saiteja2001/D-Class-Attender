# Generated by Django 4.0.6 on 2022-07-21 18:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classattendence', '0006_alter_photo_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='image',
            field=models.ImageField(upload_to='images/'),
        ),
    ]

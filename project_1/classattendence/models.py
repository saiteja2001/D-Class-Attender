from django.db import models

# Create your models here.
class Student(models.Model):
    roll_no = models.IntegerField(default= -1)
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    branch = models.CharField(max_length=100)

class Photo(models.Model):
    roll_no = models.ForeignKey(Student,on_delete=models.CASCADE, default=-1)
    image = models.ImageField(upload_to='images/')



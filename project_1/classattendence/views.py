import time
from email.mime import image
from cv2 import phase
from django.shortcuts import render,redirect
from django.http.response import StreamingHttpResponse, JsonResponse, HttpResponseServerError, HttpResponse
from django.contrib import messages
from .camera import VideoCamera
import cv2, os
import pandas as pd
import datetime

from PIL import Image,ImageTk
from.models import Student, Photo
from .forms import *
from django.core.files import File
from django.core.files.base import ContentFile
from django.core.files.temp import NamedTemporaryFile
from django.views.decorators.csrf import csrf_exempt
N_photos=50
att_rollno=[]
roll=[]
def home(request):
    return render(request , 'home.html')

def batch(request):
    val = request.POST['choice']
    if val=="new":
        fn = request.POST['firstname']
        ln = request.POST['lastname']
        if len(fn) == 0:
            messages.add_message(request, messages.INFO, 'Please enter your firstname')
            return redirect('home')
        if len(ln) == 0:
            messages.add_message(request, messages.INFO, 'Please enter your lastname')
            return redirect('home')
        try:
            r = int(request.POST['rollno'])
        except:
            messages.add_message(request, messages.INFO, 'Please enter your rollno')
            return redirect('home')
        print("----going on ---------------")
        if Student.objects.filter(roll_no=r).count()>0:
            messages.add_message(request, messages.INFO, 'Duplicate Entry')
            return redirect('home')

            
        batch_grp = request.POST['batch_grp']
        if batch_grp == "None":
            messages.add_message(request, messages.INFO, 'Please choose your branch')
            return render(request, 'home.html')
        std = Student()
        std.roll_no = r
        std.firstname = fn
        std.lastname = ln
        std.branch = batch_grp
        std.save()
        request.session['roll_no'] = std.roll_no
        messages.add_message(request, messages.INFO,", please look at the cam until it stops. At last click on upload button.")

        return render(request,'register.html')
    else:
        return render(request, 'attendence.html')
    

def register(request):
    std = Student.objects.get(roll_no = request.session['roll_no'])
    for i in range(1,N_photos+1):
        if i<11:
            photo = Photo.objects.create(roll_no = std, image = "D:\django\project_1\media\\test\{}\{}_{}.jpg".format(std.roll_no,std.roll_no,i))
            photo.save()
        else:
            photo = Photo.objects.create(roll_no=std,
                                         image="D:\django\project_1\media\\train\{}\{}_{}.jpg".format(std.roll_no,
                                                                                                     std.roll_no, i))
            photo.save()
    messages.add_message(request,messages.INFO,'registered succesfully!!!')

    return render(request, 'home.html')

def reg(request, cam):
    count = 0

    
    std = Student.objects.get(roll_no = request.session['roll_no'])
    os.makedirs("D:\django\project_1\media\\train\{}".format(std.roll_no), exist_ok=True)
    os.makedirs("D:\django\project_1\media\\test\{}".format(std.roll_no), exist_ok=True)
    while count<N_photos:
        frame,frame_byte,face = cam.get_frame()
        if face:
            count += 1

            if count<11:
                filename = 'media/test/{}/{}_{}.jpg'.format(std.roll_no,std.roll_no,count)
                try:
                    cv2.imwrite(filename,frame)

                except:
                    count -= 1
            else:
                filename = 'media/train/{}/{}_{}.jpg'.format(std.roll_no, std.roll_no, count)
                try:
                    cv2.imwrite(filename, frame)

                except:
                    count -= 1




        yield (b'--frame\r\n'
                b'Content-Type: image/jpg\r\n\r\n' + frame_byte + b'\r\n\r\n')



def register_stream(request):
    return StreamingHttpResponse((reg(request, VideoCamera())),content_type='multipart/x-mixed-replace; boundary=frame')

def att(request, cam):
    global att_rollno

    while True:
        frame,name = cam.get_name()

        if name and name not in roll:
            roll.append(name)
            std = Student.objects.get(roll_no=name)
            att_rollno.append([std.roll_no,std.firstname,std.lastname,std.branch, str(datetime.date.today()),datetime.datetime.now().time()])
        yield (b'--frame\r\n'
                b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')


                
def attendence_stream(request):
    
    return StreamingHttpResponse(att(request,VideoCamera()),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def attendence(request):
    return redirect('home')


def name_stream(request):
    global att_rollno, roll
    messages.add_message(request, messages.INFO, 'Attendance sheet downloaded!!!')
    df = pd.DataFrame(att_rollno)
    writer = pd.ExcelWriter('att.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Attendance', index=False)
    writer.save()
    roll=[]
    att_rollno=[]
    return redirect('home')

def train(request):
    messages.add_message(request, messages.INFO, 'Model Trained Successfully!!!')
    os.system('python makemodel.py')
    return redirect('/')




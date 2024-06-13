from django.http import StreamingHttpResponse, HttpResponse
from django.shortcuts import render
import cv2
import threading
from home.video_process import VideoCamera
from home.video_process_exit import VideoCamera_exit
from home import video_process, video_process_exit
from django.template import loader
from home.models import data
from django.http import JsonResponse

# Create your views here.

# def index(request):

#     emp_data = data()
#     emp_name = emp_data.name
#     enter_time = emp_data.enter_time

#     return render(request, 'index.html', context={'emp_name': emp_name, 'enter_time':enter_time})


def index(request):
    emp_data = data.objects.raw("SELECT * FROM `home_data` ORDER BY `home_data`.`id` DESC")
    doc = loader.get_template("index.html")
    context = {
    'emp_data': emp_data,
                    }
    return HttpResponse(doc.render(context, request))

def exit(request):
    emp_data = data.objects.raw("SELECT * FROM `home_data` ORDER BY `home_data`.`id` DESC")
    doc = loader.get_template("exit.html")
    context = {
    'emp_data': emp_data,
                    }
    return HttpResponse(doc.render(context, request))


def live_data(request):
    # Query the data you need
    emp_data = data.objects.raw("SELECT * FROM `home_data` ORDER BY `home_data`.`id` DESC")
    
    emp_data_list = []
    for emp in emp_data:
        emp_data_list.append({
            'id': emp.id,
            'date' : emp.date,
            'name': emp.name,
            'enter_time': emp.enter_time,
            'exit_time' : emp.exit_time,
            'exit_date' : emp.exit_date,
            # Add other fields as necessary
        })

    
    return JsonResponse({'data': emp_data_list})



# @gzip.gzip_page
def video(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(video_process.gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        return render(request, 'camnot_working.html')


# @gzip.gzip_page
def video_exit(request):
    try:
        cam = VideoCamera_exit()
        return StreamingHttpResponse(video_process_exit.gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        return render(request, 'camnot_working.html')


from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video', views.video, name='video'),
    path('video_exit', views.video_exit, name='video_exit'),
    path('exit', views.exit, name='exit'),
    path('live_data', views.live_data, name="live_data"),
]
 
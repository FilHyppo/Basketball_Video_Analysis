from django.contrib import admin
from django.urls import path
from . import views

urlpatterns=[
    path('', views.home_page, name='home_page'),
    path('upload/', views.upload_video, name='upload_video'),
    path('select_corners/<int:game_id>/', views.select_corners, name='select_corners'),
    path('save_corners/<int:game_id>/', views.save_corners, name='save_corners'),
    path('top-view/<int:game_id>/', views.top_view, name='top_view'),
]
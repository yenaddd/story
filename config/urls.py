# config/urls.py
from django.contrib import admin
from django.urls import path, include 

urlpatterns = [
    path('admin/', admin.site.urls),
    # story 앱의 URL을 /story/ 경로로 연결합니다.
    path('story/', include('story.urls')), 
]
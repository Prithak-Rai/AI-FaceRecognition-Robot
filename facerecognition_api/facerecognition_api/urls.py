"""
URL configuration for facerecognition_api project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from recognition import views

urlpatterns = [
    
    path('', views.home, name='home'),  # Home page
    path('recognize/', views.render_recognition_page, name='recognize_face_page'),  # Render the page with webcam feed
    path('recognition/recognize/', views.recognize_face, name='recognize_face'),  # Process the face recognition
    path('list_known_faces/', views.list_known_faces, name='list_known_faces'),  # List known faces
    path('add_face/', views.add_face, name='add_face'),  # Add new face to the database
    path('delete_face/', views.delete_face, name='delete_face'),  # Delete face
]

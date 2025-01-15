from django.urls import path, include
from recognition import views

urlpatterns = [
    path('', views.home, name='home'),
    path('', views.render_recognition_page, name='recognize_face_page'),  # Render the page with webcam feed
    # Route to handle face recognition
    path('recognize/', views.recognize_face, name='recognize_face'),
]

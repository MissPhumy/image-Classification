from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import  ImageClassificationModelViewSet
from . import views

router = DefaultRouter()
router.register(r'image-classification', ImageClassificationModelViewSet, basename='image-classification')

urlpatterns = [
    path('', include(router.urls)),
    path('home/', views.home, name='home'),
]


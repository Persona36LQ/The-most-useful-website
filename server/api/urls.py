from django.urls import path
from .views import ProcessImageView


urlpatterns = [
    path('image/', ProcessImageView.as_view())
]

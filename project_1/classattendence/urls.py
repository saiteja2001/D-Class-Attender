from django.urls import path

from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('batch',views.batch, name='batch'),
    path('register', views.register, name='register'),
    path('attendence_stream', views.attendence_stream, name='attendence_stream'),
    path('register_stream', views.register_stream, name='register_stream'),
    path('attendence',views.attendence, name='attendence'),
    path('name_stream', views.name_stream, name='name_stream'),
    path('train', views.train, name='train'),

]

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)
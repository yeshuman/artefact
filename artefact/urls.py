from django.contrib import admin
from django.urls import path, include
from mondos.views import mondo_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', mondo_view, name='home'),
    path('mondos/', include('mondos.urls', namespace='mondos')),
]


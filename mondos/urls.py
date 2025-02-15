from django.urls import path
from . import views

app_name = 'mondos'

urlpatterns = [
    path('', views.mondo_view, name='mondo_view'),
    path('<int:mondo_id>/', views.mondo_view, name='mondo_view'),
    path('stream/', views.mondo_stream, name='mondo_stream'),
    path('<int:mondo_id>/stream/', views.mondo_stream, name='mondo_stream'),
    path('artefacts/stream/', views.artefact_stream, name='artefact_stream'),
    path('<int:mondo_id>/artefacts/stream/', views.artefact_stream, name='artefact_stream'),
    path('message/', views.mondo_message, name='mondo_message'),
    path('artefacts/<int:entity_id>/', views.load_artefact, name='load_artefact'),
    path('<int:mondo_id>/artefacts/<int:entity_id>/', views.load_artefact, name='load_artefact'),
] 
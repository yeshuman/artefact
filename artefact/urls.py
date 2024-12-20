from django.urls import path
from mondos.views import mondo_view, mondo_stream, artefact_stream, mondo_message

urlpatterns = [
    path('', mondo_view, name='mondo_view'),
    path('stream/mondo/', mondo_stream, name='mondo_stream'),
    path('stream/artefacts/', artefact_stream, name='artefact_stream'),
    path('mondo/message/', mondo_message, name='mondo_message'),
]


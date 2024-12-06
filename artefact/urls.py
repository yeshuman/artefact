
from django.urls import path
from chats.views import chat, chat_post, chat_stream

urlpatterns = [
    path('', chat, name='chat'),
    path('chat-post/', chat_post, name='chat_post'),
    path('chat-stream/', chat_stream, name='chat_stream'),
]

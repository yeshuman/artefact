import asyncio
import json
import logging
from datetime import datetime
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from asgiref.sync import sync_to_async
from .models import Mondo, Message

logger = logging.getLogger(__name__)

def mondo_view(request):
    """Render the main conversation view."""
    return render(request, "mondos/index.html")

async def mondo_stream(request):
    """Stream Mondo messages and events."""
    response = StreamingHttpResponse(
        mondo_stream_generator(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response

async def artefact_stream(request):
    """Stream artefact detection and processing events."""
    response = StreamingHttpResponse(
        artefact_stream_generator(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response

async def mondo_stream_generator():
    """Generate SSE events for the Mondo stream."""
    i = 0
    while True:
        try:
            # Heartbeat to keep connection alive
            yield f'event: system\ndata: heartbeat {i}\n\n'
            await asyncio.sleep(1)
            i += 1
        except asyncio.CancelledError:
            logger.info(f"Mondo stream connection closed {i}")
            yield 'event: system\ndata: Connection closed\n\n'
            return

async def artefact_stream_generator():
    """Generate SSE events for the Artefact stream."""
    i = 0
    while True:
        try:
            # Heartbeat to keep connection alive
            yield f'event: system\ndata: heartbeat {i}\n\n'
            await asyncio.sleep(1)
            i += 1
        except asyncio.CancelledError:
            logger.info(f"Artefact stream connection closed {i}")
            yield 'event: system\ndata: Connection closed\n\n'
            return

@csrf_exempt
async def mondo_message(request):
    """Handle new messages in the Mondo."""
    if request.method == "POST":
        content = request.POST.get('content', '')
        mondo_id = request.POST.get('mondo_id')
        
        # Get the Mondo instance
        mondo = await sync_to_async(Mondo.objects.get)(id=mondo_id)
        
        # Create the message
        message = await Message.objects.acreate(
            mondo=mondo,
            content=content
        )
        
        return JsonResponse({"status": "success", "message_id": message.id})
    return JsonResponse({"status": "error"}, status=400)

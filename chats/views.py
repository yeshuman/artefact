import asyncio
import logging
from openai import AsyncOpenAI

from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.dispatch import receiver
from django.db.models.signals import post_save
from asgiref.sync import sync_to_async
from chats.models import HumanMessage

logger = logging.getLogger(__name__)

def chat(request):
    return render(request, "index.html")

@csrf_exempt
async def chat_post(request):
    if request.method == "POST":
        message = request.POST.get('message', '')
        await sync_to_async(HumanMessage.objects.create)(
            text=message
        )
        return JsonResponse({"status": "success"})
    return JsonResponse({"status": "error"}, status=400)

async def chat_stream(request):
    response = StreamingHttpResponse(
        chat_stream_generator(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    return response

    
latest_message = None

@receiver(post_save, sender=HumanMessage)
def handle_new_message(sender, instance, created, **kwargs):
    global latest_message
    if created:
        logger.info(f"New message: {instance.text}")
        latest_message = instance.text


async def chat_stream_generator():
    global latest_message
    client = AsyncOpenAI()
    i = 0

    while True:
        if latest_message:
            logger.info(f"Latest message: {latest_message}")
            message = latest_message
            latest_message = None  # Reset after consuming
        
            try:
                stream = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": message}
                    ],
                    stream=True
                )
                async for chunk in stream:
                    if hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            logger.info(content)
                            yield f"event: ai\ndata: <span class='chunk'>{content}</span>\n\n"
                            
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"event: system\ndata: Error: {str(e)}\n\n"

        else:
            
            try:
                # logger.info(f"heartbeat {i}")
                yield f'event: system\ndata: heartbeat {i}\n\n'
                await asyncio.sleep(1)
                i += 1  
            except asyncio.CancelledError:
                logger.info(f"Connection closed {i}")
                yield 'event: system\ndata: Connection closed\n\n'
                return

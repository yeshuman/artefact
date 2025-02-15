import asyncio
import json
import logging
from datetime import datetime
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404
from django.template.loader import render_to_string
from asgiref.sync import sync_to_async
from openai import AsyncOpenAI
from entities.services import StreamingEntityDetector
from entities.models import Entity, EntityReference, EntityArchetype
from .models import Mondo, Message, RoninMessage, SatoriMessage
from django.db import transaction
from utils.embeddings import get_embedding

logger = logging.getLogger(__name__)

# Message queue for artefact events
artefact_queues = {}

# OpenAI client for embeddings
client = AsyncOpenAI()

async def get_embedding(text: str) -> list[float]:
    """Get embedding from OpenAI."""
    response = await client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

@transaction.non_atomic_requests
async def mondo_view(request, mondo_id=None):
    """Render the main conversation view."""
    try:
        if mondo_id is None:
            # Get the latest mondo
            mondo = await sync_to_async(lambda: Mondo.objects.latest('id'))()
        else:
            mondo = await sync_to_async(get_object_or_404)(Mondo, id=mondo_id)
        
        return render(request, "mondos/mondo.html", {
            'mondo': mondo
        })
    except Mondo.DoesNotExist:
        # If no mondo exists, create a new one
        dojo = await sync_to_async(lambda: Dojo.objects.latest('id'))()
        quest = await sync_to_async(lambda: dojo.quests.latest('id'))()
        mondo = await Mondo.objects.acreate(quest=quest)
        return render(request, "mondos/mondo.html", {
            'mondo': mondo
        })

@transaction.non_atomic_requests
async def mondo_stream(request, mondo_id=None):
    """Stream Mondo messages and events."""
    try:
        if mondo_id is None:
            # Get the latest mondo
            mondo = await sync_to_async(lambda: Mondo.objects.latest('id'))()
            mondo_id = mondo.id
    except Mondo.DoesNotExist:
        # If no mondo exists, return an empty stream
        return StreamingHttpResponse(
            [],
            content_type='text/event-stream'
        )
        
    response = StreamingHttpResponse(
        mondo_stream_generator(mondo_id),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response

@transaction.non_atomic_requests
async def artefact_stream(request, mondo_id=None):
    """Stream artefact detection and processing events."""
    try:
        if mondo_id is None:
            # Get the latest mondo
            mondo = await sync_to_async(lambda: Mondo.objects.latest('id'))()
            mondo_id = mondo.id
    except Mondo.DoesNotExist:
        # If no mondo exists, return an empty stream
        return StreamingHttpResponse(
            [],
            content_type='text/event-stream'
        )
        
    queue = asyncio.Queue()
    artefact_queues[mondo_id] = queue
    
    response = StreamingHttpResponse(
        artefact_stream_generator(mondo_id, queue),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response

@transaction.non_atomic_requests
async def load_artefact(request, mondo_id, entity_id):
    """Load artefact details."""
    try:
        entity = await sync_to_async(get_object_or_404)(Entity, id=entity_id)
        html = await sync_to_async(render_to_string)(
            'mondos/partials/artefact_detail.html',
            {'entity': entity}
        )
        return HttpResponse(html)
    except Exception as e:
        logger.error(f"Error loading artefact {entity_id}: {e}")
        return HttpResponse("Error loading artefact details", status=500)

async def mondo_stream_generator(mondo_id):
    """Generate SSE events for the Mondo stream."""
    try:
        mondo = await sync_to_async(get_object_or_404)(Mondo, id=mondo_id)
        messages = mondo.messages.all()
        last_message_id = await sync_to_async(lambda: messages.last().id if messages.exists() else 0)()
        
        # Send existing messages first
        async for message in messages:
            content = await message.acontent
            author = await message.aauthor
            html = await sync_to_async(render_to_string)(
                'mondos/partials/message.html',
                {'message': {'content': content, 'author': author}}
            )
            yield f'event: message\ndata: {html}\n\n'
        
        # Then listen for new messages
        while True:
            # Check for new messages
            new_messages = mondo.messages.filter(id__gt=last_message_id)
            if await sync_to_async(new_messages.exists)():
                async for message in new_messages:
                    content = await message.acontent
                    author = await message.aauthor
                    
                    # Process content for entities if it's a new message
                    if hasattr(message, 'processed_content'):
                        content = message.processed_content
                    
                    html = await sync_to_async(render_to_string)(
                        'mondos/partials/message.html',
                        {'message': {'content': content, 'author': author}}
                    )
                    yield f'event: message\ndata: {html}\n\n'
                    last_message_id = message.id
            
            # Heartbeat
            yield 'event: heartbeat\ndata: ping\n\n'
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info(f"Mondo stream connection closed for mondo {mondo_id}")
        yield 'event: system\ndata: Connection closed\n\n'
        return

async def artefact_stream_generator(mondo_id, queue):
    """Generate SSE events for artefact detection and processing."""
    try:
        while True:
            try:
                # Wait for artefact events
                event = await asyncio.wait_for(queue.get(), timeout=30)
                
                if event['type'] == 'entity_detected':
                    # Render entity notification
                    html = await sync_to_async(render_to_string)(
                        'mondos/partials/entity.html',
                        {'entity': event['entity'], 'state': 'detected'}
                    )
                    yield f'event: entity\ndata: {json.dumps({"html": html, "id": event["entity"].id})}\n\n'
                
                elif event['type'] == 'entity_state':
                    # Send entity state update
                    yield f'event: entityState\ndata: {json.dumps({"id": event["entity_id"], "state": event["state"]})}\n\n'
                
                queue.task_done()
                
            except asyncio.TimeoutError:
                # Send heartbeat on timeout
                yield 'event: heartbeat\ndata: ping\n\n'
                
    except asyncio.CancelledError:
        logger.info(f"Artefact stream connection closed for mondo {mondo_id}")
        artefact_queues.pop(mondo_id, None)
        yield 'event: system\ndata: Connection closed\n\n'
        return

async def process_entities(message: RoninMessage) -> None:
    """Process entities in a message and update its processed content."""
    try:
        # Get or create the city archetype
        city_archetype = await sync_to_async(lambda: EntityArchetype.objects.filter(name="Location.City").first())()
        if not city_archetype:
            embedding = await get_embedding("Location.City: A major urban settlement")
            city_archetype = await EntityArchetype.objects.acreate(
                name="Location.City",
                description="A major urban settlement",
                embedding=embedding
            )
        
        # Initialize detector with city archetype
        detector = StreamingEntityDetector(mondo_id=message.mondo.id)
        detector.archetype = city_archetype
        
        # Process content in chunks to simulate streaming
        chunk_size = 100
        content = message.content
        processed_chunks = []
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            processed_chunk = await detector.process_chunk(chunk)
            processed_chunks.append(processed_chunk)
            
        # Update message with processed content
        message.processed_content = ''.join(processed_chunks)
        await message.asave()
        
    except Exception as e:
        logger.error(f"Error processing entities: {str(e)}")
        raise

async def enrich_entity(entity: Entity, mondo_id: int):
    """Enrich entity with additional data."""
    queue = artefact_queues.get(mondo_id)
    if not queue:
        return
    
    try:
        # Update state to loading
        await queue.put({
            'type': 'entity_state',
            'entity_id': entity.id,
            'state': 'loading'
        })
        
        # Get reference entity if it exists
        if entity.reference_entity_id:
            # Entity is already linked to a reference
            await queue.put({
                'type': 'entity_state',
                'entity_id': entity.id,
                'state': 'loaded'
            })
            return
            
        # Check if we should create a new reference
        if entity.is_confident:
            # Create new reference entity
            reference = await sync_to_async(EntityReference.objects.create)(
                text=entity.text,
                archetype=entity.archetype,
                embedding=entity.embedding,
                mondo_id=mondo_id
            )
            
            # Link entity to reference
            entity.reference_entity = reference
            await sync_to_async(entity.save)()
            
            await queue.put({
                'type': 'entity_state',
                'entity_id': entity.id,
                'state': 'loaded'
            })
        else:
            # Not confident enough to create reference
            await queue.put({
                'type': 'entity_state',
                'entity_id': entity.id,
                'state': 'failed'
            })
        
    except Exception as e:
        logger.error(f"Error enriching entity {entity.id}: {e}")
        if queue:
            await queue.put({
                'type': 'entity_state',
                'entity_id': entity.id,
                'state': 'failed'
            })

@csrf_exempt
@transaction.non_atomic_requests
async def mondo_message(request) -> HttpResponse:
    """Create a new message in a mondo."""
    if request.method != 'POST':
        return HttpResponse(status=405)

    try:
        data = json.loads(request.body)
        mondo_id = data.get('mondo_id')
        content = data.get('content')

        if not mondo_id or not content:
            return HttpResponse('Missing required fields', status=400)

        mondo = await sync_to_async(get_object_or_404)(Mondo, id=mondo_id)
        
        # Get the ronin object asynchronously
        quest = await sync_to_async(lambda: mondo.quest)()
        ronin = await sync_to_async(lambda: quest.ronin)()
        
        message = await RoninMessage.objects.acreate(
            mondo=mondo,
            content=content,
            author=ronin
        )

        await process_entities(message)

        return HttpResponse(status=200)
    except json.JSONDecodeError:
        return HttpResponse('Invalid JSON', status=400)
    except Exception as e:
        logger.error(f"Error creating message: {e}")
        return HttpResponse(status=500)

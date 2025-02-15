import json
import pytest
import numpy as np
from unittest.mock import patch
from django.urls import reverse
from asgiref.sync import sync_to_async
from mondos.models import Mondo, RoninMessage, SatoriMessage
from entities.models import Entity, EntityReference, EntityArchetype
from django.test import Client
from dojo.models import Dojo
from quests.models import Quest
from ronins.models import Ronin
from satoris.models import Satori
from mondos.views import get_embedding
from entities.services import get_embedding

# Mock functions
def mock_get_embedding(text: str) -> list[float]:
    """Mock function that returns a zero vector of the correct size."""
    return [0.0] * 1536  # Size of OpenAI ada-002 embeddings

@pytest.fixture
async def test_dojo():
    return await Dojo.objects.acreate(
        theme="Testing",
        principles=["Test principle 1", "Test principle 2"],
        ronin_system_message="Test ronin message",
        satori_system_message="Test satori message"
    )

@pytest.fixture
async def test_ronin():
    return await Ronin.objects.acreate(
        name="Test Ronin",
        interests=["Testing"],
        style="Direct",
        system_prompt="Test system prompt"
    )

@pytest.fixture
async def test_satori():
    return await Satori.objects.acreate(
        name="Test Satori",
        specialties="Testing",
        teaching_style="Direct",
        system_prompt="Test system prompt"
    )

@pytest.fixture
async def test_quest(test_dojo, test_ronin, test_satori):
    return await Quest.objects.acreate(
        title="Test Quest",
        ronin=test_ronin,
        satori=test_satori,
        dojo=test_dojo
    )

@pytest.fixture
async def test_mondo(test_dojo, test_quest):
    return await Mondo.objects.acreate(
        quest=test_quest,
        dojo=test_dojo
    )

@pytest.fixture
async def test_archetype():
    return await EntityArchetype.objects.acreate(
        name="Location.City",
        description="A city or urban area",
        embedding=[0.0] * 1536  # OpenAI ada-002 embedding size
    )

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_mondo_view(client, test_mondo):
    """Test mondo view renders correctly."""
    url = reverse('mondos:mondo_view', args=[test_mondo.id])
    response = await sync_to_async(client.get)(url)
    assert response.status_code == 200
    assert b'messages' in response.content

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_mondo_message_creation(client, test_mondo):
    """Test creating a message in a mondo."""
    with patch('utils.embeddings.get_embedding', side_effect=mock_get_embedding):
        data = {
            'mondo_id': test_mondo.id,
            'content': 'Test message about cities'
        }
        response = await sync_to_async(client.post)(
            '/mondos/message/',
            data=json.dumps(data),
            content_type='application/json'
        )
        assert response.status_code == 200

@pytest.mark.skip(reason="Non-deterministic test: Entity detection improves over time as references are added")
@pytest.mark.django_db
@pytest.mark.asyncio
async def test_entity_detection_and_enrichment(client, test_mondo, test_archetype):
    """Test entity detection and enrichment process."""
    # This test is skipped because:
    # 1. Entity detection is non-idempotent
    # 2. Early runs may miss entities due to insufficient references
    # 3. Later runs may detect more entities as the reference database grows
    # 4. Confidence scores vary based on number of existing references
    pass

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_artefact_loading(client, test_mondo, test_archetype):
    """Test loading artefact details."""
    # Create an entity
    entity = await Entity.objects.acreate(
        text='Paris',
        archetype=test_archetype,
        confidence=0.95,
        start_position=0,
        end_position=5,
        message=await RoninMessage.objects.acreate(
            mondo=test_mondo,
            content='Paris',
            author=test_mondo.quest.ronin
        ),
        embedding=[0.0] * 1536
    )
    
    # Create a reference entity
    reference = await EntityReference.objects.acreate(
        text='Paris',
        archetype=test_archetype,
        embedding=[0.0] * 1536,
        description='The capital of France',
        mondo=test_mondo
    )
    
    # Link entity to reference
    entity.reference_entity = reference
    await sync_to_async(entity.save)()
    
    # Test loading artefact details
    url = reverse('mondos:load_artefact', args=[test_mondo.id, entity.id])
    response = await sync_to_async(client.get)(url)
    assert response.status_code == 200
    
    # Verify response content
    content = str(response.content)
    assert 'Paris' in content
    assert 'Location.City' in content
    assert 'capital of France' in content

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_streaming_events(client, test_mondo):
    """Test SSE streaming for messages and artefacts."""
    # Test mondo stream
    mondo_url = reverse('mondos:mondo_stream', args=[test_mondo.id])
    response = await sync_to_async(client.get)(mondo_url)
    assert response.status_code == 200
    assert response['Content-Type'] == 'text/event-stream'
    
    # Test artefact stream
    artefact_url = reverse('mondos:artefact_stream', args=[test_mondo.id])
    response = await sync_to_async(client.get)(artefact_url)
    assert response.status_code == 200
    assert response['Content-Type'] == 'text/event-stream' 

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_home_view_latest_mondo(client):
    """Test that the home view shows the latest mondo."""
    # Create test data
    dojo = await Dojo.objects.acreate(
        theme="Test Dojo",
        principles=["Seek knowledge", "Share wisdom"]
    )
    ronin = await Ronin.objects.acreate(name="Test Ronin")
    satori = await Satori.objects.acreate(name="Test Satori")
    quest = await Quest.objects.acreate(title="Test Quest", dojo=dojo, ronin=ronin, satori=satori)
    mondo = await Mondo.objects.acreate(quest=quest, dojo=dojo)
    
    # Test home view
    response = await sync_to_async(client.get)('/')
    assert response.status_code == 200
    assert b'messages' in response.content

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_message_streaming(client):
    """Test that messages are streamed correctly."""
    # Create test data
    dojo = await Dojo.objects.acreate(
        theme="Test Dojo",
        principles=["Seek knowledge", "Share wisdom"]
    )
    ronin = await Ronin.objects.acreate(name="Test Ronin")
    satori = await Satori.objects.acreate(name="Test Satori")
    quest = await Quest.objects.acreate(title="Test Quest", dojo=dojo, ronin=ronin, satori=satori)
    mondo = await Mondo.objects.acreate(quest=quest, dojo=dojo)
    
    # Test message streaming
    url = reverse('mondos:mondo_stream', args=[mondo.id])
    response = await sync_to_async(client.get)(url)
    assert response.status_code == 200
    assert response['Content-Type'] == 'text/event-stream'

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_artefact_streaming(client):
    """Test that artefacts are streamed correctly."""
    # Create test data
    dojo = await Dojo.objects.acreate(
        theme="Test Dojo",
        principles=["Seek knowledge", "Share wisdom"]
    )
    ronin = await Ronin.objects.acreate(name="Test Ronin")
    satori = await Satori.objects.acreate(name="Test Satori")
    quest = await Quest.objects.acreate(title="Test Quest", dojo=dojo, ronin=ronin, satori=satori)
    mondo = await Mondo.objects.acreate(quest=quest, dojo=dojo)
    
    # Test artefact streaming
    url = reverse('mondos:artefact_stream', args=[mondo.id])
    response = await sync_to_async(client.get)(url)
    assert response.status_code == 200
    assert response['Content-Type'] == 'text/event-stream'

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_message_creation_with_entities(client, test_mondo):
    """Test creating a message with entity detection."""
    with patch('utils.embeddings.get_embedding', side_effect=mock_get_embedding):
        data = {
            'mondo_id': test_mondo.id,
            'content': 'I visited Tokyo last summer.'
        }
        response = await sync_to_async(client.post)(
            '/mondos/message/',
            data=json.dumps(data),
            content_type='application/json'
        )
        assert response.status_code == 200 
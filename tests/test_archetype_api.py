import pytest
from django.test import TransactionTestCase
from django.db import transaction
from openai import AsyncOpenAI
from entities.models import EntityArchetype, EntityReference, Entity
from entities.services import StreamingEntityDetector
from mondos.models import Mondo, RoninMessage
from dojo.models import Dojo
from quests.models import Quest
from ronins.models import Ronin
from satoris.models import Satori
from asgiref.sync import sync_to_async
from functools import partial
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_atomic():
    """Async context manager for database transactions."""
    async with sync_to_async(transaction.atomic)():
        yield

@pytest.fixture
async def openai_client():
    """Create a real OpenAI client for testing."""
    return AsyncOpenAI()

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
        satori=test_satori
    )

@pytest.fixture
async def test_mondo(test_dojo, test_quest):
    return await Mondo.objects.acreate(
        dojo=test_dojo,
        quest=test_quest
    )

@pytest.fixture
async def test_message(test_mondo, test_ronin):
    return await RoninMessage.objects.acreate(
        mondo=test_mondo,
        content="Test message about Paris and London",
        author=test_ronin
    )

@pytest.fixture
async def location_archetype(openai_client):
    """Create a location archetype for testing."""
    description = "Geographic locations like cities, countries, and landmarks"
    embedding = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=description
    )
    
    return await EntityArchetype.objects.acreate(
        name="Location.Geographic",
        description=description,
        embedding=embedding.data[0].embedding
    )

@pytest.mark.django_db(transaction=True)
@pytest.mark.real
async def test_entity_detection_with_real_embeddings(openai_client, test_message, location_archetype):
    """Test detecting entities using real embeddings and similarity matching."""
    # Create reference for Paris
    paris_embedding = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input="Paris"
    )
    await EntityReference.objects.acreate(
        text="Paris",
        archetype=location_archetype,
        mondo=test_message.mondo,
        embedding=paris_embedding.data[0].embedding
    )
    
    # Create a detector
    detector = StreamingEntityDetector(mondo_id=test_message.mondo.id)
    detector.archetype = location_archetype
    
    # Test detection in text with a similar but non-exact location
    text = "I would love to visit the French capital someday."
    
    # Create a new message for this test
    message = await RoninMessage.objects.acreate(
        mondo=test_message.mondo,
        content=text,
        author=test_message.author
    )
    await sync_to_async(message.refresh_from_db)()  # Ensure message is properly saved
    
    async def get_embedding(text):
        """Async wrapper for getting embeddings."""
        embedding = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return embedding.data[0].embedding
    
    # First detection with single reference
    marked_text, entities = await detector.process_chunk(text, message.id, get_embedding)
    
    # Verify basic detection works
    capital_entity = next((e for e in entities if "capital" in e["text"].lower()), None)
    assert capital_entity is not None, "Should detect 'capital' as a location-related term"
    initial_confidence = capital_entity["confidence"]
    
    # Add more reference entities to improve detection
    additional_locations = ["London", "Tokyo", "New York", "French capital"]
    for location in additional_locations:
        location_embedding = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=location
        )
        await EntityReference.objects.acreate(
            text=location,
            archetype=location_archetype,
            mondo=test_message.mondo,
            embedding=location_embedding.data[0].embedding
        )
    
    # Test detection with enriched references
    marked_text, entities = await detector.process_chunk(text, message.id, get_embedding)
    capital_entity = next((e for e in entities if "capital" in e["text"].lower()), None)
    assert capital_entity is not None, "Should still detect 'capital' as a location"
    enriched_confidence = capital_entity["confidence"]
    
    # Verify confidence improves with more references
    assert enriched_confidence > initial_confidence, "Confidence should improve with more references"
    
    # Test non-location text
    non_location_text = "The happy dog jumped over the fence."
    non_location_message = await RoninMessage.objects.acreate(
        mondo=test_message.mondo,
        content=non_location_text,
        author=test_message.author
    )
    marked_text, entities = await detector.process_chunk(non_location_text, non_location_message.id, get_embedding)
    
    # If any entities were detected in non-location text, verify their confidence is lower
    if entities:
        non_location_confidence = max(e["confidence"] for e in entities)
        assert enriched_confidence > non_location_confidence, "Location entities should have higher confidence than non-location matches"

@pytest.mark.django_db(transaction=True)
@pytest.mark.real
async def test_context_aware_entity_detection(openai_client, test_message, location_archetype):
    """Test entity detection with context awareness."""
    # Create initial reference for Paris
    paris_embedding = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input="Paris"
    )
    await EntityReference.objects.acreate(
        text="Paris",
        archetype=location_archetype,
        mondo=test_message.mondo,
        embedding=paris_embedding.data[0].embedding
    )
    
    # Create a detector
    detector = StreamingEntityDetector(mondo_id=test_message.mondo.id)
    detector.archetype = location_archetype
    
    # Test detection with initial reference
    text = "While in the city of lights, I visited a famous museum. The museum was amazing."
    
    # Create a new message for this test
    message = await RoninMessage.objects.acreate(
        mondo=test_message.mondo,
        content=text,
        author=test_message.author
    )
    await sync_to_async(message.refresh_from_db)()  # Ensure message is properly saved
    
    async def get_embedding(text):
        """Async wrapper for getting embeddings."""
        embedding = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return embedding.data[0].embedding
    
    # First detection with single reference
    marked_text, entities = await detector.process_chunk(text, message.id, get_embedding)
    
    # Verify basic detection works
    city_entity = next((e for e in entities if "city" in e["text"].lower()), None)
    assert city_entity is not None, "Should detect 'city of lights' as a location"
    initial_city_confidence = city_entity["confidence"]
    
    # Add more contextual references
    for ref_text in ["City of Lights", "Louvre Museum", "French capital"]:
        ref_embedding = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=ref_text
        )
        await EntityReference.objects.acreate(
            text=ref_text,
            archetype=location_archetype,
            mondo=test_message.mondo,
            embedding=ref_embedding.data[0].embedding
        )
    
    # Test detection with enriched references
    marked_text, entities = await detector.process_chunk(text, message.id, get_embedding)
    
    # Verify entities are detected with improved confidence
    city_entity = next((e for e in entities if "city" in e["text"].lower()), None)
    museum_entity = next((e for e in entities if "museum" in e["text"].lower()), None)
    
    assert city_entity is not None, "Should still detect 'city of lights' as a location"
    assert museum_entity is not None, "Should detect 'museum' as a location"
    
    # Verify confidence improved with context
    assert city_entity["confidence"] > initial_city_confidence, "City confidence should improve with context"
    
    # Test non-location text
    non_location_text = "The cat and dog played together."
    non_location_message = await RoninMessage.objects.acreate(
        mondo=test_message.mondo,
        content=non_location_text,
        author=test_message.author
    )
    marked_text, entities = await detector.process_chunk(non_location_text, non_location_message.id, get_embedding)
    
    # If any entities were detected in non-location text, verify their confidence is lower
    if entities:
        non_location_confidence = max(e["confidence"] for e in entities)
        assert city_entity["confidence"] > non_location_confidence, "Location entities should have higher confidence than non-location matches" 
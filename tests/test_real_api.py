import pytest
import numpy as np
from openai import AsyncOpenAI
from entities.models import Entity, EntityArchetype, EntityReference
from entities.services import StreamingEntityDetector
from mondos.models import Mondo, RoninMessage
from dojo.models import Dojo
from quests.models import Quest
from ronins.models import Ronin
from satoris.models import Satori
import os

@pytest.fixture
async def test_dojo():
    return await Dojo.objects.acreate(
        theme="Testing Real API",
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
        content="I would love to visit Paris and explore the Louvre museum. The city's rich history and cultural heritage make it a fascinating destination.",
        author=test_ronin
    )

@pytest.fixture
async def openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment")
    return AsyncOpenAI(api_key=api_key)

@pytest.mark.real
@pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason="OpenAI API key not configured"
)
@pytest.mark.django_db(transaction=True)
async def test_create_location_archetype(openai_client):
    """Test creating a location archetype with real embeddings."""
    # Clean up existing archetypes and references
    await EntityReference.objects.all().adelete()
    await Entity.objects.all().adelete()
    await EntityArchetype.objects.all().adelete()
    
    # Create location archetype with real embedding
    description = "Cities, countries, and geographic locations"
    embedding_response = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=f"location: {description}"
    )
    embedding = embedding_response.data[0].embedding
    
    location_archetype = await EntityArchetype.objects.acreate(
        name="location",
        description=description,
        embedding=embedding
    )
    
    # Verify archetype
    assert location_archetype is not None
    assert location_archetype.name == "location"
    assert location_archetype.description == description
    assert len(location_archetype.embedding) == 1536
    
    # Create some example locations for reference
    locations = ["Paris", "London", "Tokyo"]
    for location in locations:
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=location
        )
        embedding = embedding_response.data[0].embedding
        
        await EntityReference.objects.acreate(
            text=location,
            archetype=location_archetype,
            embedding=embedding
        )
    
    # Verify references were created
    references = await EntityReference.objects.filter(archetype=location_archetype).acount()
    assert references == len(locations)

@pytest.mark.real
@pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason="OpenAI API key not configured"
)
@pytest.mark.django_db(transaction=True)
async def test_entity_detection_with_real_embeddings(openai_client, test_message):
    """Test entity detection using real embeddings."""
    # Clean up existing entities
    await Entity.objects.all().adelete()

    # Create location archetype if it doesn't exist
    location_archetype = await EntityArchetype.objects.filter(name="location").afirst()
    if not location_archetype:
        description = "Cities, countries, and geographic locations"
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=f"location: {description}"
        )
        embedding = embedding_response.data[0].embedding

        location_archetype = await EntityArchetype.objects.acreate(
            name="location",
            description=description,
            embedding=embedding
        )

    # Create reference for Paris if it doesn't exist
    paris_ref = await EntityReference.objects.filter(text="Paris").afirst()
    if not paris_ref:
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input="Paris"
        )
        embedding = embedding_response.data[0].embedding

        paris_ref = await EntityReference.objects.acreate(
            text="Paris",
            archetype=location_archetype,
            embedding=embedding,
            mondo=test_message.mondo
        )

    # Initialize entity detector
    detector = StreamingEntityDetector(mondo_id=test_message.mondo.id)
    detector.archetype = location_archetype

    # Create async embedding function using real OpenAI API
    async def get_embedding(text: str) -> np.ndarray:
        response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    # Process text with real embeddings
    text = test_message.content
    marked_text, entities = await detector.process_chunk(text, test_message.id, get_embedding)

    # Verify that Paris was detected
    assert len(entities) > 0
    paris_entity = next((e for e in entities if e["text"].lower() == "paris"), None)
    assert paris_entity is not None
    assert paris_entity["type"] == "location"
    assert paris_entity["confidence"] > 0.9

@pytest.mark.real
@pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason="OpenAI API key not configured"
)
@pytest.mark.django_db(transaction=True)
async def test_context_aware_entity_detection(openai_client, test_message):
    """Test context-aware entity detection with real embeddings."""
    # Clean up existing entities
    await Entity.objects.all().adelete()
    
    # Create museum archetype
    description = "Museums, galleries, and cultural institutions"
    embedding_response = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=f"museum: {description}"
    )
    embedding = embedding_response.data[0].embedding
    
    museum_archetype = await EntityArchetype.objects.acreate(
        name="museum",
        description=description,
        embedding=embedding
    )
    
    # Create Louvre reference
    embedding_response = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input="Louvre museum"
    )
    embedding = embedding_response.data[0].embedding
    
    louvre_ref = await EntityReference.objects.acreate(
        text="Louvre",
        archetype=museum_archetype,
        embedding=embedding,
        mondo=test_message.mondo
    )
    
    # Initialize entity detector
    detector = StreamingEntityDetector(mondo_id=test_message.mondo.id)
    detector.archetype = museum_archetype
    
    # Create async embedding function using real OpenAI API
    async def get_embedding(text: str) -> np.ndarray:
        response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    # Process text with real embeddings
    text = test_message.content
    marked_text, entities = await detector.process_chunk(text, test_message.id, get_embedding)
    
    # Verify entity detection
    assert len(entities) >= 1
    louvre_entity = next((e for e in entities if e["text"] == "Louvre"), None)
    assert louvre_entity is not None
    assert louvre_entity["type"] == "museum"
    assert louvre_entity["confidence"] > 0.5
    assert "Louvre" in marked_text
    assert "<entity" in marked_text and "</entity>" in marked_text 
import pytest
from openai import AsyncOpenAI
from entities.models import EntityArchetype, EntityReference, Entity
from entities.services import StreamingEntityDetector
from mondos.models import Mondo, RoninMessage
from dojo.models import Dojo
from quests.models import Quest
from ronins.models import Ronin
from satoris.models import Satori

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

@pytest.mark.django_db(transaction=True)
@pytest.mark.real
async def test_create_location_archetype(openai_client, test_message):
    """Test creating a location archetype with real embeddings."""
    # Create a location archetype
    description = "Geographic locations like cities, countries, and landmarks"
    embedding = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=description
    )
    
    archetype = await EntityArchetype.objects.acreate(
        name="Location.Geographic",
        description=description,
        embedding=embedding.data[0].embedding
    )
    
    # Verify the archetype
    assert archetype.name == "Location.Geographic"
    assert len(archetype.embedding) == 1536  # OpenAI embedding size
    
    # Create some reference entities
    locations = ["Paris", "London", "Mount Everest", "Louvre Museum"]
    for location in locations:
        embedding = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=location
        )
        await EntityReference.objects.acreate(
            text=location,
            archetype=archetype,
            mondo=test_message.mondo,
            embedding=embedding.data[0].embedding
        )
    
    # Verify references were created
    refs = await EntityReference.objects.filter(archetype=archetype).acount()
    assert refs == len(locations)

@pytest.mark.django_db(transaction=True)
@pytest.mark.real
async def test_entity_detection_with_real_embeddings(openai_client, test_message):
    """Test detecting entities using real embeddings and similarity matching."""
    # Create a location archetype first
    description = "Geographic locations like cities, countries, and landmarks"
    embedding = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=description
    )
    
    archetype = await EntityArchetype.objects.acreate(
        name="Location.Geographic",
        description=description,
        embedding=embedding.data[0].embedding
    )
    
    # Create reference for Paris
    paris_embedding = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input="Paris"
    )
    await EntityReference.objects.acreate(
        text="Paris",
        archetype=archetype,
        mondo=test_message.mondo,
        embedding=paris_embedding.data[0].embedding
    )
    
    # Create a detector
    detector = StreamingEntityDetector(mondo_id=test_message.mondo.id)
    detector.archetype = archetype
    
    # Test detection in text
    text = "I would love to visit Paris someday."
    
    def get_embedding(text):
        """Sync wrapper for getting embeddings."""
        return paris_embedding.data[0].embedding  # Use same embedding for testing
    
    marked_text, entities = await detector.process_chunk(text, test_message.id, get_embedding)
    
    # Verify detection
    assert len(entities) == 1
    assert entities[0]["text"] == "Paris"
    assert entities[0]["type"] == "Location.Geographic"
    assert entities[0]["confidence"] > 0.9  # Should be very confident for exact match
    assert "<entity" in marked_text and "Paris" in marked_text

@pytest.mark.django_db(transaction=True)
@pytest.mark.real
async def test_context_aware_entity_detection(openai_client, test_message):
    """Test entity detection with context awareness."""
    # Create a location archetype
    description = "Geographic locations like cities, countries, and landmarks"
    embedding = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=description
    )
    
    archetype = await EntityArchetype.objects.acreate(
        name="Location.Geographic",
        description=description,
        embedding=embedding.data[0].embedding
    )
    
    # Create reference for Paris and Louvre
    for place in ["Paris", "Louvre Museum"]:
        place_embedding = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=place
        )
        await EntityReference.objects.acreate(
            text=place,
            archetype=archetype,
            mondo=test_message.mondo,
            embedding=place_embedding.data[0].embedding
        )
    
    # Create a detector
    detector = StreamingEntityDetector(mondo_id=test_message.mondo.id)
    detector.archetype = archetype
    
    # Test detection with context
    text = "While in Paris, I visited the Louvre. The museum was amazing."
    
    async def get_embedding(text):
        """Async wrapper for getting embeddings."""
        embedding = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return embedding.data[0].embedding
    
    marked_text, entities = await detector.process_chunk(text, test_message.id, get_embedding)
    
    # Verify both entities were detected
    assert len(entities) == 2
    paris_entity = next(e for e in entities if e["text"] == "Paris")
    louvre_entity = next(e for e in entities if e["text"] == "Louvre")
    
    assert paris_entity["type"] == "Location.Geographic"
    assert louvre_entity["type"] == "Location.Geographic"
    assert paris_entity["confidence"] > 0.8
    assert louvre_entity["confidence"] > 0.8 
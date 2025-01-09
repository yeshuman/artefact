import pytest
import numpy as np
from django.conf import settings
from entities.models import Entity, EntityReference, EntityArchetype
from entities.services import StreamingEntityDetector
from mondos.models import Mondo
from dojo.models import Dojo
from quests.models import Quest
from ronins.models import Ronin
from satoris.models import Satori

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
    from mondos.models import RoninMessage
    return await RoninMessage.objects.acreate(
        mondo=test_mondo,
        content="Test message about Paris and London",
        author=test_ronin
    )

@pytest.fixture
async def location_archetype():
    # Create a vector of ones for the archetype
    ones_vector = np.ones(1536, dtype=np.float32)
    
    # Try to get existing archetype first
    archetype = await EntityArchetype.objects.filter(name="location").afirst()
    if archetype:
        return archetype
        
    # Create new if doesn't exist
    return await EntityArchetype.objects.acreate(
        name="location",
        description="A place or location",
        embedding=ones_vector.tolist()  # Convert to list for JSON serialization
    )

@pytest.fixture
async def paris_reference(location_archetype, test_mondo):
    return await EntityReference.objects.acreate(
        text="Paris",
        archetype=location_archetype,
        mondo=test_mondo,
        embedding=np.ones(1536, dtype=np.float32)  # Use ones vector for reference
    )

@pytest.fixture
async def london_reference(location_archetype, test_mondo):
    return await EntityReference.objects.acreate(
        text="London",
        archetype=location_archetype,
        mondo=test_mondo,
        embedding=np.ones(1536, dtype=np.float32)  # Use ones vector for reference
    )

@pytest.fixture
def mock_embedding_fn():
    """Create a mock embedding function that returns matching vectors for known entities."""
    async def _mock_embedding(text: str) -> np.ndarray:
        # Return ones vector for known entities to match reference embeddings
        if text.lower() in ["paris", "london"]:
            return np.ones(1536, dtype=np.float32)
        # Return zeros for other words to ensure no false matches
        return np.zeros(1536, dtype=np.float32)
    return _mock_embedding

@pytest.fixture
def detector(test_mondo):
    """Create a detector instance for testing."""
    detector = StreamingEntityDetector(mondo_id=test_mondo.id)
    detector.similarity_threshold = 0.5  # Lower threshold for testing
    return detector

@pytest.mark.django_db(transaction=True)
async def test_streaming_entity_detector_word_boundary(
    detector, paris_reference, location_archetype, mock_embedding_fn, test_message
):
    """Test that entities are detected at word boundaries."""
    detector.archetype = location_archetype  # Set the archetype
    text = "I love Paris in the springtime."
    marked_text, entities = await detector.process_chunk(text, test_message.id, mock_embedding_fn)
    
    # Verify that Paris was detected
    assert len(entities) == 1
    assert entities[0]["text"] == "Paris"
    assert entities[0]["type"] == "location"
    assert entities[0]["confidence"] > 0.9

@pytest.mark.django_db(transaction=True)
async def test_streaming_entity_detector_context_window(
    detector, paris_reference, location_archetype, mock_embedding_fn, test_message
):
    """Test that context window is maintained correctly."""
    detector.archetype = location_archetype  # Set the archetype
    text1 = "I love"
    text2 = "Paris in"
    text3 = "the springtime."
    
    marked1, entities1 = await detector.process_chunk(text1, test_message.id, mock_embedding_fn)
    marked2, entities2 = await detector.process_chunk(text2, test_message.id, mock_embedding_fn)
    marked3, entities3 = await detector.process_chunk(text3, test_message.id, mock_embedding_fn)
    
    # Verify that Paris was detected in the second chunk
    assert len(entities2) == 1
    assert entities2[0]["text"] == "Paris"
    assert entities2[0]["type"] == "location"
    assert entities2[0]["confidence"] > 0.9

@pytest.mark.django_db(transaction=True)
async def test_streaming_entity_detector_markup(
    detector, paris_reference, london_reference, location_archetype, mock_embedding_fn, test_message
):
    """Test that entity markup is added correctly for multiple entities."""
    detector.archetype = location_archetype  # Set the archetype
    text = "I went from Paris to London yesterday."
    marked_text, entities = await detector.process_chunk(text, test_message.id, mock_embedding_fn)
    
    assert len(entities) == 2
    paris_entity = next(e for e in entities if e["text"] == "Paris")
    london_entity = next(e for e in entities if e["text"] == "London")
    
    assert paris_entity["type"] == location_archetype.name
    assert london_entity["type"] == location_archetype.name
    assert paris_entity["confidence"] == pytest.approx(1.0, abs=1e-6)
    assert london_entity["confidence"] == pytest.approx(1.0, abs=1e-6)
    
    expected_text = (
        f'I went from <entity id="{paris_entity["id"]}" type="{location_archetype.name}">Paris</entity> to '
        f'<entity id="{london_entity["id"]}" type="{location_archetype.name}">London</entity> yesterday.'
    )
    assert marked_text == expected_text

@pytest.mark.asyncio
@pytest.mark.mock
@pytest.mark.django_db(transaction=True)
async def test_entity_archetype_creation():
    """Test that entity archetypes are created without duplicates."""
    from entities.models import EntityArchetype
    
    # Ensure we start with a clean slate
    await EntityArchetype.objects.all().adelete()
    
    # Verify we start with a clean slate
    initial_count = await EntityArchetype.objects.acount()
    assert initial_count == 0, "Expected empty database"
    
    # Create initial archetype
    embedding = np.zeros(1536, dtype=np.float32)
    concept_archetype = await EntityArchetype.objects.acreate(
        name="concept",
        description="Abstract ideas or principles",
        embedding=embedding
    )
    
    # Verify first archetype
    assert concept_archetype is not None
    assert concept_archetype.description == "Abstract ideas or principles"
    assert len(concept_archetype.embedding) == 1536
    
    # Check for duplicate name
    is_duplicate = await EntityArchetype.is_duplicate("concept", embedding)
    assert is_duplicate, "Should detect duplicate name"
    
    # Check for similar embedding with different name
    similar_embedding = np.zeros(1536, dtype=np.float32)  # Same embedding
    is_duplicate = await EntityArchetype.is_duplicate("different_name", similar_embedding)
    assert is_duplicate, "Should detect similar embedding"
    
    # Create different archetype with different embedding
    different_embedding = np.ones(1536, dtype=np.float32)  # Different embedding
    technique_archetype = await EntityArchetype.objects.acreate(
        name="technique",
        description="Specific methods or practices",
        embedding=different_embedding
    )
    
    # Verify both archetypes exist
    archetypes = await EntityArchetype.objects.acount()
    assert archetypes == 2, "Expected 2 unique archetypes" 

@pytest.mark.django_db(transaction=True)
async def test_high_confidence_reference_creation(
    detector, location_archetype, mock_embedding_fn, test_message
):
    """Test that high confidence matches are stored as reference entities."""
    detector.archetype = location_archetype

    # Create initial references for Paris and London
    paris_ref = await EntityReference.objects.acreate(
        text="Paris",
        archetype=location_archetype,
        mondo_id=detector.mondo_id,
        embedding=np.ones(1536, dtype=np.float32)
    )
    london_ref = await EntityReference.objects.acreate(
        text="London",
        archetype=location_archetype,
        mondo_id=detector.mondo_id,
        embedding=np.ones(1536, dtype=np.float32)
    )

    # Process text with both locations
    text = "I visited Paris and London."
    marked_text, entities = await detector.process_chunk(text, test_message.id, mock_embedding_fn)

    # Verify both entities were detected
    assert len(entities) == 2
    paris_entity = next(e for e in entities if e["text"] == "Paris")
    london_entity = next(e for e in entities if e["text"] == "London")

    assert paris_entity["type"] == location_archetype.name
    assert london_entity["type"] == location_archetype.name
    assert paris_entity["confidence"] == pytest.approx(1.0, abs=1e-6)
    assert london_entity["confidence"] == pytest.approx(1.0, abs=1e-6) 

@pytest.mark.django_db(transaction=True)
async def test_reference_entity_storage_threshold(
    detector, location_archetype, mock_embedding_fn, test_message
):
    """Test that only high confidence matches are stored as reference entities."""
    detector.archetype = location_archetype
    
    # Create initial reference for Paris
    paris_ref = await EntityReference.objects.acreate(
        text="Paris",
        archetype=location_archetype,
        mondo_id=detector.mondo_id,
        embedding=np.ones(1536, dtype=np.float32)
    )
    
    # Process text with a high confidence match (Paris) and a lower confidence match
    text = "I visited Paris and Berlin."
    marked_text, entities = await detector.process_chunk(text, test_message.id, mock_embedding_fn)
    
    # Verify both entities were detected
    assert len(entities) == 1  # Only Paris should be detected
    paris_entity = entities[0]
    assert paris_entity["text"] == "Paris"
    assert paris_entity["confidence"] == pytest.approx(1.0, abs=1e-6)
    
    # Verify no new reference was created for Berlin (lower confidence)
    berlin_refs = await EntityReference.objects.filter(
        text="Berlin",
        archetype=location_archetype,
        mondo_id=detector.mondo_id
    ).acount()
    assert berlin_refs == 0

@pytest.mark.django_db(transaction=True)
async def test_multi_word_entity_detection(
    detector, location_archetype, test_message
):
    """Test detection of multi-word entities."""
    detector.archetype = location_archetype
    
    # Create reference for multi-word location
    new_york_ref = await EntityReference.objects.acreate(
        text="New York",
        archetype=location_archetype,
        mondo_id=detector.mondo_id,
        embedding=np.ones(1536, dtype=np.float32)
    )
    
    # Create custom embedding function for multi-word test
    async def multi_word_embedding_fn(text: str) -> np.ndarray:
        if text.lower() in ["new york", "new", "york"]:
            return np.ones(1536, dtype=np.float32)
        return np.zeros(1536, dtype=np.float32)
    
    # Process text with multi-word entity
    text = "I love New York City."
    marked_text, entities = await detector.process_chunk(text, test_message.id, multi_word_embedding_fn)
    
    # Verify entity was detected
    assert len(entities) == 1
    ny_entity = entities[0]
    assert ny_entity["text"] == "New York"
    assert ny_entity["type"] == location_archetype.name
    assert ny_entity["confidence"] == pytest.approx(1.0, abs=1e-6)
    
    # Verify markup
    expected_text = (
        f'I love <entity id="{ny_entity["id"]}" type="{location_archetype.name}">'
        f'New York</entity> City.'
    )
    assert marked_text == expected_text 
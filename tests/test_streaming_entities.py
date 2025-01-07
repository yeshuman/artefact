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
async def test_message(db, test_mondo, test_ronin):
    from mondos.models import RoninMessage
    message = await RoninMessage.objects.acreate(
        mondo=test_mondo,
        content="Test message about Paris and London",
        author=test_ronin
    )
    return message

@pytest.fixture
async def location_archetype():
    archetype = await EntityArchetype.objects.acreate(
        name="location",
        description="A place or location",
        embedding=np.zeros(1536)  # Default zero vector for testing
    )
    return archetype

@pytest.fixture
async def paris_reference(location_archetype, test_mondo):
    return await EntityReference.objects.acreate(
        text="Paris",
        type="location",
        mondo=test_mondo,
        embedding=np.zeros(1536)  # Default zero vector for testing
    )

@pytest.fixture
async def london_reference(location_archetype, test_mondo):
    return await EntityReference.objects.acreate(
        text="London",
        type="location", 
        mondo=test_mondo,
        embedding=np.zeros(1536)  # Default zero vector for testing
    )

@pytest.fixture
def mock_embedding_fn():
    def _mock_embedding(text: str) -> np.ndarray:
        return np.zeros(1536)  # Return zero vector for testing
    return _mock_embedding

@pytest.fixture
def detector(test_mondo):
    return StreamingEntityDetector(mondo_id=test_mondo.id)

@pytest.mark.django_db
async def test_streaming_entity_detector_word_boundary(
    detector, paris_reference, location_archetype, mock_embedding_fn
):
    """Test that entities are detected at word boundaries."""
    detector.archetype = location_archetype  # Set the archetype
    text = "I love Paris in the springtime."
    marked_text, entities = await detector.process_chunk(text, 1, mock_embedding_fn)
    
    assert len(entities) == 1
    assert entities[0]["text"] == "Paris"
    assert entities[0]["type"] == "location"
    assert entities[0]["confidence"] == 1.0
    assert marked_text == f'I love <entity id="{entities[0]["id"]}">Paris</entity> in the springtime.'

@pytest.mark.django_db
async def test_streaming_entity_detector_context_window(
    detector, paris_reference, location_archetype, mock_embedding_fn
):
    """Test that context window is maintained correctly."""
    detector.archetype = location_archetype  # Set the archetype
    text1 = "I love"
    text2 = "Paris in"
    text3 = "the springtime."
    
    marked1, entities1 = await detector.process_chunk(text1, 1, mock_embedding_fn)
    assert len(entities1) == 0
    assert marked1 == text1
    
    marked2, entities2 = await detector.process_chunk(text2, 1, mock_embedding_fn)
    assert len(entities2) == 1
    assert entities2[0]["text"] == "Paris"
    assert marked2 == f'<entity id="{entities2[0]["id"]}">Paris</entity> in'
    
    marked3, entities3 = await detector.process_chunk(text3, 1, mock_embedding_fn)
    assert len(entities3) == 0
    assert marked3 == text3

@pytest.mark.django_db
async def test_streaming_entity_detector_markup(
    detector, paris_reference, london_reference, location_archetype, mock_embedding_fn
):
    """Test that entity markup is added correctly for multiple entities."""
    detector.archetype = location_archetype  # Set the archetype
    text = "I went from Paris to London yesterday."
    marked_text, entities = await detector.process_chunk(text, 1, mock_embedding_fn)
    
    assert len(entities) == 2
    paris_entity = next(e for e in entities if e["text"] == "Paris")
    london_entity = next(e for e in entities if e["text"] == "London")
    
    assert paris_entity["type"] == "location"
    assert london_entity["type"] == "location"
    assert paris_entity["confidence"] == 1.0
    assert london_entity["confidence"] == 1.0
    
    expected_text = (
        f'I went from <entity id="{paris_entity["id"]}">Paris</entity> to '
        f'<entity id="{london_entity["id"]}">London</entity> yesterday.'
    )
    assert marked_text == expected_text 
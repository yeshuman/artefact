import pytest
import pytest_asyncio
from typing import AsyncGenerator
from asgiref.sync import sync_to_async
from django.conf import settings

pytestmark = [pytest.mark.django_db]

@pytest.fixture
def sample_text_chunks():
    """Sample text chunks containing potential entities."""
    return [
        "The ancient temples of Kyoto offer a glimpse into Japan's past.",
        "Mount Fuji stands as a symbol of Japanese culture.",
        "The bustling streets of Tokyo's Shibuya district never sleep.",
        "In the serene gardens of Ryoan-ji Temple, one finds peace.",
        "The historic Gion district preserves Kyoto's geisha traditions."
    ]

@pytest.fixture
def expected_entities():
    """Expected entities and their types from the sample texts."""
    return [
        {
            'text': 'Kyoto',
            'type': 'PLACE',
            'confidence_threshold': 0.8
        },
        {
            'text': 'Mount Fuji',
            'type': 'LANDMARK',
            'confidence_threshold': 0.9
        },
        {
            'text': "Shibuya",
            'type': 'PLACE',
            'confidence_threshold': 0.8
        },
        {
            'text': 'Ryoan-ji Temple',
            'type': 'SITE',
            'confidence_threshold': 0.85
        },
        {
            'text': 'Gion',
            'type': 'SITE',
            'confidence_threshold': 0.8
        }
    ]

@pytest_asyncio.fixture
async def mondo_with_messages(prepared_mock_dojo):
    """Create a mondo with sample messages."""
    mondo = await prepared_mock_dojo.mondo()
    
    # Add sample messages
    for chunk in sample_text_chunks():
        await mondo.messages.acreate(
            content=chunk,
            author=prepared_mock_dojo.satori.model_obj
        )
    
    return mondo

@pytest.mark.asyncio
async def test_quick_pattern_matching():
    """Test the initial quick pattern matching for potential entities."""
    from entities.detection import quick_pattern_match
    
    # Test single chunk
    chunk = "The ancient temples of Kyoto offer a glimpse into Japan's past."
    matches = await quick_pattern_match(chunk)
    
    assert len(matches) > 0
    assert any(m.text == "Kyoto" for m in matches)
    assert all(m.start >= 0 and m.end <= len(chunk) for m in matches)

@pytest.mark.asyncio
async def test_vector_similarity():
    """Test vector similarity matching for entity validation."""
    from entities.detection import vector_similarity
    
    # Test known place name
    confidence = await vector_similarity("Kyoto")
    assert confidence > settings.ENTITY_CONFIDENCE_THRESHOLD
    
    # Test non-place text
    confidence = await vector_similarity("random text")
    assert confidence < settings.ENTITY_CONFIDENCE_THRESHOLD

@pytest.mark.asyncio
async def test_entity_detection_in_message():
    """Test complete entity detection process in a message."""
    from entities.detection import detect_entities
    from mondos.models import Message
    
    # Create test message
    message = await Message.objects.acreate(
        content="The ancient temples of Kyoto offer a glimpse into Japan's past."
    )
    
    # Detect entities
    entities = await detect_entities(message)
    
    assert len(entities) > 0
    entity = entities[0]
    assert entity.text == "Kyoto"
    assert entity.type == "PLACE"
    assert entity.confidence > settings.ENTITY_CONFIDENCE_THRESHOLD
    assert entity.message == message

@pytest.mark.asyncio
async def test_entity_position_tracking():
    """Test that entity positions in text are correctly tracked."""
    from entities.detection import detect_entities
    from mondos.models import Message
    
    content = "Mount Fuji stands as a symbol of Japanese culture."
    message = await Message.objects.acreate(content=content)
    
    entities = await detect_entities(message)
    assert len(entities) > 0
    
    entity = entities[0]
    assert entity.text == "Mount Fuji"
    assert content[entity.start_position:entity.end_position] == "Mount Fuji"

@pytest.mark.asyncio
async def test_duplicate_entity_handling():
    """Test how repeated entities in text are handled."""
    from entities.detection import detect_entities
    from mondos.models import Message
    
    content = "Kyoto's temples are magnificent. Kyoto's gardens are peaceful."
    message = await Message.objects.acreate(content=content)
    
    entities = await detect_entities(message)
    
    # Should detect Kyoto twice, with different positions
    kyoto_entities = [e for e in entities if e.text == "Kyoto"]
    assert len(kyoto_entities) == 2
    assert kyoto_entities[0].start_position != kyoto_entities[1].start_position

@pytest.mark.asyncio
async def test_entity_confidence_thresholds():
    """Test that confidence thresholds are properly applied."""
    from entities.detection import detect_entities
    from mondos.models import Message
    
    # Test with high-confidence entity
    message = await Message.objects.acreate(
        content="Mount Fuji stands majestic."
    )
    entities = await detect_entities(message)
    assert any(e.text == "Mount Fuji" and e.confidence > 0.9 for e in entities)
    
    # Test with low-confidence potential entity
    message = await Message.objects.acreate(
        content="The mountain stands majestic."
    )
    entities = await detect_entities(message)
    assert not any(e.text == "mountain" for e in entities)

@pytest.mark.asyncio
async def test_entity_type_classification():
    """Test that entities are correctly classified by type."""
    from entities.detection import detect_entities
    from mondos.models import Message
    
    test_cases = [
        ("Kyoto is beautiful.", "Kyoto", "PLACE"),
        ("Mount Fuji is majestic.", "Mount Fuji", "LANDMARK"),
        ("Ryoan-ji Temple is peaceful.", "Ryoan-ji Temple", "SITE")
    ]
    
    for content, expected_text, expected_type in test_cases:
        message = await Message.objects.acreate(content=content)
        entities = await detect_entities(message)
        
        matching_entities = [e for e in entities if e.text == expected_text]
        assert len(matching_entities) == 1
        assert matching_entities[0].type == expected_type 
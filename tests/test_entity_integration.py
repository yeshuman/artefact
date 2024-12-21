import pytest
import pytest_asyncio
from asgiref.sync import sync_to_async
from mondos.models import Message
from entities.models import Entity

pytestmark = [pytest.mark.django_db]

@pytest_asyncio.fixture
async def sample_message(prepared_mock_dojo):
    """Create a sample message with known entities."""
    mondo = await prepared_mock_dojo.mondo()
    message = await Message.objects.acreate(
        content="The ancient temples of Kyoto and Mount Fuji are iconic Japanese landmarks.",
        mondo=mondo,
        author=prepared_mock_dojo.satori.model_obj
    )
    return message

@pytest.mark.asyncio
async def test_entity_detection_on_message_creation(sample_message):
    """Test that entities are detected when a message is created."""
    # Get entities for the message
    entities = await sync_to_async(list)(Entity.objects.filter(message=sample_message))
    
    # Verify entities were detected
    assert len(entities) >= 2
    
    # Check for specific entities
    entity_texts = [e.text for e in entities]
    assert "Kyoto" in entity_texts
    assert "Mount Fuji" in entity_texts
    
    # Verify entity types
    for entity in entities:
        if entity.text == "Kyoto":
            assert entity.type == Entity.PLACE
        elif entity.text == "Mount Fuji":
            assert entity.type == Entity.LANDMARK

@pytest.mark.asyncio
async def test_entity_confidence_filtering(prepared_mock_dojo):
    """Test that low-confidence entities are filtered out."""
    mondo = await prepared_mock_dojo.mondo()
    
    # Create message with a low-confidence potential entity
    message = await Message.objects.acreate(
        content="The mountain was beautiful.",
        mondo=mondo,
        author=prepared_mock_dojo.satori.model_obj
    )
    
    # Verify no entities were created (confidence too low)
    entity_count = await sync_to_async(Entity.objects.filter(message=message).count)()
    assert entity_count == 0

@pytest.mark.asyncio
async def test_multiple_entity_occurrences(prepared_mock_dojo):
    """Test handling of repeated entities in the same message."""
    mondo = await prepared_mock_dojo.mondo()
    
    # Create message with repeated entities
    message = await Message.objects.acreate(
        content="Kyoto's temples are beautiful. Kyoto's gardens are peaceful.",
        mondo=mondo,
        author=prepared_mock_dojo.satori.model_obj
    )
    
    # Get Kyoto entities
    kyoto_entities = await sync_to_async(list)(
        Entity.objects.filter(message=message, text="Kyoto")
    )
    
    # Should detect both occurrences
    assert len(kyoto_entities) == 2
    
    # Verify different positions
    positions = [(e.start_position, e.end_position) for e in kyoto_entities]
    assert len(set(positions)) == 2  # Unique positions

@pytest.mark.asyncio
async def test_entity_detection_in_conversation_flow(prepared_mock_dojo):
    """Test entity detection through a conversation flow."""
    mondo = await prepared_mock_dojo.mondo()
    
    # Simulate a conversation with multiple messages
    messages = []
    for content in [
        "Let's discuss the temples of Kyoto.",
        "Mount Fuji is visible from Tokyo on clear days.",
        "The Gion district preserves traditional culture."
    ]:
        message = await Message.objects.acreate(
            content=content,
            mondo=mondo,
            author=prepared_mock_dojo.satori.model_obj
        )
        messages.append(message)
    
    # Verify entities across all messages
    all_entities = await sync_to_async(list)(Entity.objects.filter(message__mondo=mondo))
    
    # Check total entities
    assert len(all_entities) >= 4
    
    # Verify unique entities
    unique_entities = {e.text for e in all_entities}
    expected_entities = {"Kyoto", "Mount Fuji", "Tokyo", "Gion"}
    assert expected_entities.issubset(unique_entities)
    
    # Check entity types are consistent
    entity_types = {e.text: e.type for e in all_entities}
    assert all(entity_types[e.text] == e.type for e in all_entities if e.text in entity_types) 
import pytest
from django.conf import settings
from entities.models import EntityArchetype, EntityReference, Entity
from mondos.models import Message, Mondo
from dojo.models import Dojo
from quests.models import Quest
from ronins.models import Ronin
from satoris.models import Satori
from asgiref.sync import sync_to_async
import numpy as np

@pytest.fixture
async def location_archetype(db):
    """Create a Location.Geographic archetype for testing."""
    return await EntityArchetype.objects.acreate(
        name="Geographic",
        description="Cities, countries, and other geographic locations",
        embedding=np.random.rand(1536).tolist()  # Mock embedding for testing
    )

@pytest.fixture
async def test_dojo(db):
    """Create a test dojo for mondo creation."""
    return await Dojo.objects.acreate(
        theme="Test Dojo Theme",
        principles=["Test Principle 1", "Test Principle 2"],
        ronin_system_message="Test Ronin Message",
        satori_system_message="Test Satori Message"
    )

@pytest.fixture
async def test_ronin(db):
    """Create a test ronin for quest creation."""
    return await Ronin.objects.acreate(
        name="Test Ronin",
        system_prompt="Test system prompt",
        meditation_prompt="Test meditation prompt"
    )

@pytest.fixture
async def test_satori(db):
    """Create a test satori for quest creation."""
    return await Satori.objects.acreate(
        name="Test Satori",
        system_prompt="Test system prompt",
        meditation_prompt="Test meditation prompt"
    )

@pytest.fixture
async def test_quest(db, test_dojo, test_ronin, test_satori):
    """Create a test quest for mondo creation."""
    return await Quest.objects.acreate(
        title="Test Quest",
        ronin=test_ronin,
        satori=test_satori
    )

@pytest.fixture
async def test_mondo(db, test_dojo, test_quest):
    """Create a test mondo for reference entities."""
    return await Mondo.objects.acreate(
        dojo=test_dojo,
        quest=test_quest
    )

@pytest.fixture
async def test_message(db, test_mondo):
    """Create a test message containing entities to detect."""
    return await Message.objects.acreate(
        mondo=test_mondo,
        content="I would love to visit Paris and explore the Louvre museum."
    )

@pytest.fixture
async def paris_reference(db, test_mondo, location_archetype):
    """Create a reference entity for Paris."""
    return await EntityReference.objects.acreate(
        text="Paris",
        archetype=location_archetype,
        embedding=np.random.rand(1536).tolist(),  # Mock embedding
        mondo=test_mondo
    )

@pytest.mark.asyncio
async def test_create_entity_archetype(db):
    """Test creation of entity archetypes with hierarchical structure."""
    # Create parent archetype
    location = await EntityArchetype.objects.acreate(
        name="Location",
        description="Physical locations and places",
        embedding=np.random.rand(1536).tolist()
    )
    
    # Create child archetype
    city = await EntityArchetype.objects.acreate(
        name="City",
        description="Urban settlements",
        parent=location,
        embedding=np.random.rand(1536).tolist()
    )
    
    # Verify hierarchy
    assert city.parent == location
    assert await location.children.acount() == 1
    assert city.full_path == "Location.City"

@pytest.mark.asyncio
async def test_entity_detection_basic(db, location_archetype, test_message, paris_reference):
    """Test basic entity detection in message text."""
    # Create detected entity
    entity = await Entity.objects.acreate(
        text="Paris",
        archetype=location_archetype,
        confidence=0.95,
        start_position=20,  # Position in test message
        end_position=25,
        message=test_message,
        reference_entity=paris_reference,
        embedding=np.random.rand(1536).tolist()
    )
    
    # Verify entity properties
    assert entity.text == "Paris"
    assert entity.is_confident
    assert entity.length == 5
    assert entity.archetype == location_archetype
    assert entity.reference_entity == paris_reference

@pytest.mark.asyncio
async def test_entity_confidence_threshold(db, location_archetype, test_message):
    """Test entity confidence threshold behavior."""
    # Create entities with different confidence scores
    high_conf = await Entity.objects.acreate(
        text="Paris",
        archetype=location_archetype,
        confidence=0.9,
        start_position=20,
        end_position=25,
        message=test_message,
        embedding=np.random.rand(1536).tolist()
    )
    
    low_conf = await Entity.objects.acreate(
        text="Louvre",
        archetype=location_archetype,
        confidence=0.3,
        start_position=41,
        end_position=47,
        message=test_message,
        embedding=np.random.rand(1536).tolist()
    )
    
    # Test confidence threshold
    assert high_conf.is_confident
    assert not low_conf.is_confident

@pytest.mark.asyncio
async def test_vector_similarity_search(db, test_mondo, location_archetype):
    """Test vector similarity search for entity matching."""
    # Create reference entities with known embeddings
    base_embedding = np.random.rand(1536)
    similar_embedding = base_embedding + np.random.normal(0, 0.1, 1536)
    different_embedding = np.random.rand(1536)
    
    ref1 = await EntityReference.objects.acreate(
        text="Tokyo",
        archetype=location_archetype,
        embedding=base_embedding.tolist(),
        mondo=test_mondo
    )
    
    ref2 = await EntityReference.objects.acreate(
        text="Tokyo, Japan",
        archetype=location_archetype,
        embedding=similar_embedding.tolist(),
        mondo=test_mondo
    )
    
    ref3 = await EntityReference.objects.acreate(
        text="London",
        archetype=location_archetype,
        embedding=different_embedding.tolist(),
        mondo=test_mondo
    )
    
    # Query similar entities using L2 distance
    raw_query = """
        SELECT *, embedding <-> %s::vector as distance
        FROM entities_entityreference
        ORDER BY distance
        LIMIT 3
    """
    
    # Convert raw query to async
    similar_refs = await sync_to_async(list)(
        EntityReference.objects.raw(raw_query, [base_embedding.tolist()])
    )
    
    # Verify results
    assert len(similar_refs) >= 2  # Should find at least two results
    
    # Get distances for verification
    distances = [getattr(ref, 'distance', float('inf')) for ref in similar_refs]
    
    # First result should be the exact match (distance ≈ 0)
    assert distances[0] < 0.1  # Some small threshold for floating point comparison
    
    # Second result should be the similar vector
    assert distances[1] < distances[2]  # Each subsequent result should be further away 
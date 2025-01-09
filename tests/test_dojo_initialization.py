import pytest
from unittest.mock import patch, AsyncMock
from dojo.dojo import Dojo as DojoService
from dojo.models import Dojo
from entities.models import Entity, EntityArchetype, EntityReference
from tests.test_dojo import state, MockOpenAI
import json
from asgiref.sync import sync_to_async

@pytest.fixture
async def mock_llm_client():
    """Mock OpenAI client for testing."""
    return MockOpenAI()

@pytest.fixture
async def mock_dojo(mock_llm_client):
    """Create a test dojo with a mock client."""
    return DojoService(llm_client=mock_llm_client)

@pytest.fixture
def mock_responses():
    """Mock responses for the dojo initialization."""
    return [
        # Response for dojo initialization
        json.dumps({
            "theme": "Test Dojo Theme",
            "principles": ["Test Principle 1", "Test Principle 2"],
            "ronin_system_message": "Test Ronin System Message",
            "satori_system_message": "Test Satori System Message"
        }),
        # Response for archetype creation
        json.dumps([{
            "name": "test_archetype",
            "description": "Test archetype description",
            "examples": ["example1", "example2"]
        }])
    ]

@pytest.mark.django_db(transaction=True)
async def test_dojo_initialization(mock_dojo, mock_responses):
    """Test that a dojo can be initialized with the correct theme and principles."""
    # Set up mock responses
    state['responses'] = mock_responses
    state['message_count'] = 0
    
    # Initialize the dojo
    await mock_dojo.initialize()
    
    # Verify dojo was created with correct attributes
    dojo = await Dojo.objects.aget()
    assert dojo is not None
    assert dojo.theme == "Test Dojo Theme"
    assert dojo.principles == ["Test Principle 1", "Test Principle 2"]
    assert dojo.ronin_system_message == "Test Ronin System Message"
    assert dojo.satori_system_message == "Test Satori System Message"
    
    # Verify archetypes were created
    archetypes = [archetype async for archetype in EntityArchetype.objects.all()]
    assert len(archetypes) == 1
    assert archetypes[0].name == "test_archetype"
    assert archetypes[0].description == "Test archetype description"
    
    # Verify reference entities were created
    references = [ref async for ref in EntityReference.objects.all()]
    assert len(references) == 2
    assert references[0].text == "example1"
    assert references[1].text == "example2"
    
    # Get archetype field asynchronously
    get_archetype = sync_to_async(lambda ref: ref.archetype)
    ref_archetypes = [await get_archetype(ref) for ref in references]
    assert all(arch == archetypes[0] for arch in ref_archetypes) 
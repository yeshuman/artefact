import pytest
import numpy as np
from django.conf import settings
from entities.models import EntityArchetype, EntityReference, Entity
from mondos.models import Message, Mondo
from dojo.models import Dojo
from quests.models import Quest
from ronins.models import Ronin
from satoris.models import Satori
from satoris.satori import Satori as SatoriService
from asgiref.sync import sync_to_async
import os
import sys

@pytest.fixture
async def test_dojo(db):
    """Create a test dojo for testing."""
    return await Dojo.objects.acreate(
        theme="Test Dojo Theme",
        principles=["Test Principle 1", "Test Principle 2"],
        ronin_system_message="Test Ronin Message",
        satori_system_message="Test Satori Message"
    )

@pytest.fixture
async def test_ronin(db):
    """Create a test ronin for testing."""
    return await Ronin.objects.acreate(
        name="Test Ronin",
        system_prompt="Test system prompt",
        meditation_prompt="Test meditation prompt"
    )

@pytest.fixture
async def test_satori(db):
    """Create a test satori for testing."""
    return await Satori.objects.acreate(
        name="Test Satori",
        system_prompt="Test system prompt",
        model_name="gpt-4-1106-preview"
    )

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    class MockLLMClient:
        async def embeddings_create(self, input, model="text-embedding-ada-002"):
            embedding = np.random.rand(1536).tolist()  # Mock embedding
            return type('Response', (), {
                'data': [type('Data', (), {
                    'embedding': embedding
                })]
            })
            
        async def chat_completions_create(self, model, messages, temperature, stream):
            class MockStream:
                def __init__(self):
                    self.responses = [
                        "Wisdom is a profound ",
                        "understanding that comes ",
                        "through experience and reflection."
                    ]
                    self.index = 0
                
                async def __aiter__(self):
                    return self
                
                async def __anext__(self):
                    if self.index >= len(self.responses):
                        raise StopAsyncIteration
                    
                    response = type('Response', (), {})
                    response.choices = [type('Choice', (), {})]
                    response.choices[0].delta = type('Delta', (), {
                        'content': self.responses[self.index]
                    })
                    self.index += 1
                    return response

            return MockStream()
        
        chat = type('Chat', (), {'completions': type('Completions', (), {'create': chat_completions_create})()})()
        embeddings = type('Embeddings', (), {'create': embeddings_create})()
    
    return MockLLMClient()

@pytest.mark.asyncio
async def test_satori_entity_detection_mock(
    db, test_dojo, test_ronin, test_satori, mock_llm_client
):
    """Test that Satori can detect entities in its responses."""
    
    # Create mondo and message
    test_mondo = await Mondo.objects.acreate(
        dojo=test_dojo,
        ronin=test_ronin
    )
    
    test_message = await Message.objects.acreate(
        mondo=test_mondo,
        content="What is the nature of wisdom?",
        author="ronin"
    )

    # Create concept archetype
    concept_archetype = await EntityArchetype.objects.acreate(
        name="Concept",
        description="Abstract concepts and ideas",
        embedding=np.random.rand(1536).tolist()
    )

    # Create reference entity
    wisdom_ref = await EntityReference.objects.acreate(
        text="wisdom",
        archetype=concept_archetype,
        embedding=np.random.rand(1536).tolist()
    )

    # Initialize Satori service
    satori_service = SatoriService(
        name=test_satori.name,
        model_instance=test_satori,
        llm_client=mock_llm_client
    )

    # Get Satori's response
    response = await satori_service.respond(test_mondo, test_message)

    # Verify entity detection
    entities = await Entity.objects.filter(message=response).acount()
    assert entities > 0, "Should detect entities in Satori's response"

@pytest.mark.asyncio
async def test_satori_entity_detection_real(
    db, test_dojo, test_ronin, test_satori, mock_llm_client
):
    """Test entity detection in Satori's responses using real embeddings."""
    
    # Create mondo and message
    test_mondo = await Mondo.objects.acreate(
        dojo=test_dojo,
        ronin=test_ronin
    )
    
    test_message = await Message.objects.acreate(
        mondo=test_mondo,
        content="Tell me about the meditation halls at Mount Fuji.",
        author="ronin"
    )

    # Create location archetype
    location_archetype = await EntityArchetype.objects.acreate(
        name="Location",
        description="Geographic locations and places",
        embedding=np.random.rand(1536).tolist()
    )

    # Create reference entities
    mount_fuji_ref = await EntityReference.objects.acreate(
        text="Mount Fuji",
        archetype=location_archetype,
        embedding=np.random.rand(1536).tolist()
    )

    meditation_hall_ref = await EntityReference.objects.acreate(
        text="meditation hall",
        archetype=location_archetype,
        embedding=np.random.rand(1536).tolist()
    )

    # Initialize Satori service
    satori_service = SatoriService(
        name=test_satori.name,
        model_instance=test_satori,
        llm_client=mock_llm_client
    )

    # Get Satori's response
    response = await satori_service.respond(test_mondo, test_message)

    # Verify entity detection
    entities = await Entity.objects.filter(message=response).acount()
    assert entities > 0, "Should detect location entities in Satori's response" 
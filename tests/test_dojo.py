import json
import pytest
import logging
from unittest.mock import AsyncMock, patch
from dojo.models import Dojo
from dojo.dojo import Dojo as DojoService
from entities.models import Entity, EntityArchetype, EntityReference
from ronins.models import Ronin
from satoris.models import Satori
from asgiref.sync import sync_to_async
from django.db import transaction

logger = logging.getLogger(__name__)

# Initialize state for mock responses
state = {
    'message_count': 0,
    'exchange_count': 0,
    'last_speaker': None,
    'responses': None  # Will be set from the fixture
}

class MockStreamResponse:
    def __init__(self, content):
        self.content = content
        self.current_pos = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.current_pos >= len(self.content):
            raise StopAsyncIteration
        
        chunk = self.content[self.current_pos:self.current_pos + 1]
        self.current_pos += 1
        
        return type('StreamChoice', (), {
            'choices': [type('Choice', (), {
                'delta': type('Delta', (), {
                    'content': chunk
                })
            })]
        })

class MockAsyncCompletions:
    @classmethod
    async def create(cls, **kwargs):
        # Get the prompt content
        system_content = kwargs['messages'][0]['content']
        user_content = kwargs['messages'][-1]['content'] if len(kwargs['messages']) > 1 else None
        
        # Increment message count
        state['message_count'] += 1
        
        # Get appropriate response based on message count
        response = state['responses'][state['message_count'] - 1]
        
        # Return mock stream
        return MockStreamResponse(response)

class MockChat:
    completions = MockAsyncCompletions()

class MockEmbeddings:
    @classmethod
    async def create(cls, **kwargs):
        # Return mock embedding of size 1536
        return type('Response', (), {
            'data': [type('Data', (), {
                'embedding': [0.0] * 1536
            })]
        })

class MockOpenAI:
    chat = MockChat()
    embeddings = MockEmbeddings()

@pytest.mark.asyncio
@pytest.mark.mock
@pytest.mark.django_db(transaction=True)
async def test_mondo_qa_interaction_mock(mock_llm_responses):
    """Test basic Q&A interaction in a Mondo."""
    # Reset state
    state['message_count'] = 0
    state['exchange_count'] = 0
    state['last_speaker'] = None
    state['responses'] = [
        # Response for Ronin prompts
        json.dumps({
            "system_prompt": "Test Ronin System Prompt",
            "meditation_prompt": "Test Ronin Meditation Prompt"
        }),
        # Response for Ronin meditation
        json.dumps({
            "name": "Test Ronin Name",
            "interests": ["Test Interest"],
            "style": "Test Style"
        }),
        # Response for Satori prompts
        json.dumps({
            "system_prompt": "Test Satori System Prompt",
            "meditation_prompt": "Test Satori Meditation Prompt"
        }),
        # Response for Satori meditation
        json.dumps({
            "name": "Test Satori Name",
            "specialties": ["Test Specialty"],
            "teaching_style": "Test Teaching Style"
        }),
        # Response for Ronin quest contemplation
        json.dumps({
            "quest_title": "Test Quest Title",
            "description": "Test quest description"
        }),
        # Response for initial question
        json.dumps({
            "content": "Test initial response",
            "type": "answer"
        }),
        # Response for follow-up question
        json.dumps({
            "content": "Test follow-up response",
            "type": "answer"
        }),
        # Response for initial Ronin message
        json.dumps({
            "content": "Test initial response",
            "type": "answer"
        }),
        # Response for follow-up Ronin message
        json.dumps({
            "content": "Test follow-up response",
            "type": "answer"
        })
    ]
    
    # Create test dojo with mock client
    mock_client = MockOpenAI()
    dojo_model = await Dojo.objects.acreate(
        theme="Test Dojo",
        principles=["Test principle"],
        ronin_system_message="Test ronin message",
        satori_system_message="Test satori message"
    )
    
    # Create test ronin
    ronin = await Ronin.objects.acreate(
        name="Test Ronin",
        interests=["Testing"],
        style="Direct",
        system_prompt="Test system prompt"
    )
    
    # Create test satori
    satori = await Satori.objects.acreate(
        name="Test Satori",
        specialties=["Testing"],
        teaching_style="Direct",
        system_prompt="Test system prompt"
    )
    
    # Create dojo service
    dojo_service = DojoService(llm_client=mock_client)
    dojo_service.model = dojo_model
    
    # Prepare Ronin and Satori
    dojo_service.ronin_obj = ronin
    dojo_service.satori_obj = satori
    await dojo_service.prepare_ronin()
    await dojo_service.prepare_satori()
    
    # Create mondo
    mondo = await dojo_service.mondo()
    
    # Create initial question message
    from mondos.models import RoninMessage
    question = "What is the meaning of life?"
    question_message = await RoninMessage.objects.acreate(
        mondo=mondo,
        content=question,
        author=ronin
    )
    
    # Get Satori service to respond
    from satoris.satori import Satori as SatoriService
    satori_service = SatoriService(
        name=satori.name,
        specialties=satori.specialties,
        teaching_style=satori.teaching_style,
        model_obj=satori,
        llm_client=mock_client
    )
    
    # Get response
    response = await satori_service.respond(question_message)
    
    # Verify response
    assert response is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    
    # Test follow-up question
    follow_up = "Can you elaborate on that?"
    follow_up_message = await RoninMessage.objects.acreate(
        mondo=mondo,
        content=follow_up,
        author=ronin
    )
    
    # Get follow-up response
    response = await satori_service.respond(follow_up_message)
    
    # Verify follow-up response
    assert response is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0

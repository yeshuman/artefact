import json
import pytest
import logging
from unittest.mock import AsyncMock, patch

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
        
        # Determine which response to use based on the prompt content
        response_content = None
        
        if "Create a unique philosophical framework for a Dojo" in system_content:
            response_content = json.dumps(state['responses']['dojo_contemplation'])
            logger.debug(f"Returning dojo contemplation response: {response_content}")
        elif "Through deep meditation, discover your identity as a seeker" in (user_content or ''):
            # Ronin meditation response
            response_content = json.dumps({
                'name': state['responses']['ronin_meditation']['name'],
                'interests': state['responses']['ronin_meditation']['interests'],
                'style': state['responses']['ronin_meditation']['style']
            })
            logger.debug(f"Returning ronin meditation response: {response_content}")
        elif "Through deep meditation, discover your identity as a guide" in (user_content or ''):
            # Satori meditation response
            response_content = json.dumps({
                'name': state['responses']['satori_meditation']['name'],
                'specialties': state['responses']['satori_meditation']['specialties'],
                'teaching_style': state['responses']['satori_meditation']['teaching_style']
            })
            logger.debug(f"Returning satori meditation response: {response_content}")
        elif "Name this quest" in (user_content or ''):
            response_content = json.dumps(state['responses']['quest_naming'])
        elif "Through meditation, envision the quest" in system_content:
            response_content = json.dumps(state['responses']['quest_contemplation'])
        elif "Generate a thoughtful opening question" in (user_content or ''):
            response_content = state['responses']['shomon']
            state['message_count'] += 1
            state['last_speaker'] = 'ronin'
            logger.debug(f"Message {state['message_count']}: Shomon question")
        elif "Have you reached a state of understanding" in system_content:
            # Only end after one exchange (3 messages total)
            should_end = state['exchange_count'] >= 1
            response_content = json.dumps({
                'should_end': should_end,
                'reason': state['responses']['understanding_contemplation'][0]['reason']
            })
            logger.debug(f"Understanding check: exchanges={state['exchange_count']}, should_end={should_end}")
        # Check if this is a Satori response based on their system prompt
        elif "guide who illuminates through subtle direction" in system_content:
            # Satori's response
            state['message_count'] += 1
            state['last_speaker'] = 'satori'
            response_content = state['responses']['satori_answer']
            logger.debug(f"Message {state['message_count']}: Satori answers (exchange {state['exchange_count']})")
        # Check if this is a Ronin response based on their system prompt
        elif "seeker of wisdom, walking the path between knowledge and understanding" in system_content:
            # Ronin's response
            state['message_count'] += 1
            state['last_speaker'] = 'ronin'
            response_content = state['responses']['ronin_question']
            state['exchange_count'] += 1
            logger.debug(f"Message {state['message_count']}: Ronin asks question (exchange {state['exchange_count']})")
        else:
            # Default to continuing the conversation based on last speaker
            if state['last_speaker'] == 'ronin':
                state['message_count'] += 1
                state['last_speaker'] = 'satori'
                response_content = state['responses']['satori_answer']
                logger.debug(f"Message {state['message_count']}: Satori answers (exchange {state['exchange_count']})")
            else:
                state['message_count'] += 1
                state['last_speaker'] = 'ronin'
                response_content = state['responses']['ronin_question']
                state['exchange_count'] += 1
                logger.debug(f"Message {state['message_count']}: Ronin asks question (exchange {state['exchange_count']})")
        
        logger.debug(f"Current state - Messages: {state['message_count']}, Exchanges: {state['exchange_count']}, Last speaker: {state['last_speaker']}")
        
        # Return streamed response if streaming is requested
        if kwargs.get('stream', False):
            return MockStreamResponse(response_content)
        
        # Return regular response
        return type('Response', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {
                    'content': response_content
                })
            })]
        })

class MockChat:
    completions = MockAsyncCompletions()

class MockOpenAI:
    chat = MockChat()

@pytest.mark.asyncio
@pytest.mark.mock
@pytest.mark.django_db
async def test_mondo_qa_interaction_mock(mock_llm_responses):
    """Test the Q&A interaction between Ronin and Satori."""
    # Reset state
    state['message_count'] = 0
    state['exchange_count'] = 0
    state['last_speaker'] = None
    state['responses'] = mock_llm_responses
    
    with patch('openai.AsyncOpenAI', return_value=MockOpenAI()):
        from dojo.dojo import Dojo
        from ronins.ronin import Ronin
        from satoris.satori import Satori
        
        # Create a new dojo
        dojo = Dojo(llm_client=MockOpenAI())
        await dojo.initialize()
        
        # Create and prepare Ronin
        ronin = await dojo.prepare_ronin(
            style=mock_llm_responses['ronin_meditation']['style']
        )
        
        # Create and prepare Satori
        satori = await dojo.prepare_satori(
            teaching_style=mock_llm_responses['satori_meditation']['teaching_style']
        )
        
        # Start the mondo
        mondo = await dojo.mondo()
        messages = [msg async for msg in mondo.messages.all()]
        assert len(messages) == 1
        assert messages[0].content == mock_llm_responses['shomon']
        
        # Continue the mondo for one exchange
        await dojo.continue_mondo(mondo)
        messages = [msg async for msg in mondo.messages.all()]
        assert len(messages) == 3  # Initial + 2 new messages
        assert messages[0].content == mock_llm_responses['shomon']  # Initial question
        assert messages[1].content == mock_llm_responses['satori_answer']  # First answer
        assert messages[2].content == mock_llm_responses['ronin_question']  # Follow-up

@pytest.mark.asyncio
@pytest.mark.mock
@pytest.mark.django_db
async def test_mondo_continue_conversation_unlimited_mock(mock_llm_responses):
    """Test continuing a mondo conversation until understanding is reached."""
    # Reset state
    state['message_count'] = 0
    state['exchange_count'] = 0
    state['last_speaker'] = None
    state['responses'] = mock_llm_responses
    
    with patch('openai.AsyncOpenAI', return_value=MockOpenAI()):
        from dojo.dojo import Dojo
        from ronins.ronin import Ronin
        from satoris.satori import Satori
        
        # Create a new dojo
        dojo = Dojo(llm_client=MockOpenAI())
        await dojo.initialize()
        
        # Create and prepare Ronin
        ronin = await dojo.prepare_ronin(
            style=mock_llm_responses['ronin_meditation']['style']
        )
        
        # Create and prepare Satori
        satori = await dojo.prepare_satori(
            teaching_style=mock_llm_responses['satori_meditation']['teaching_style']
        )
        
        # Start the mondo
        mondo = await dojo.mondo()
        messages = [msg async for msg in mondo.messages.all()]
        assert len(messages) == 1
        assert messages[0].content == mock_llm_responses['shomon']
        
        # Continue the mondo until understanding is reached
        while True:
            await dojo.continue_mondo(mondo)
            messages = [msg async for msg in mondo.messages.all()]
            
            # Check if understanding is reached
            understanding = await ronin.contemplate_understanding(messages[-1].content)
            if understanding['should_end']:
                break
        
        # Verify we have all expected messages (3 total):
        # 1. Initial question (shomon)
        # 2. Satori's first answer
        # 3. Ronin's follow-up question
        messages = [msg async for msg in mondo.messages.all()]
        assert len(messages) == 3
        
        # Verify message contents
        assert messages[0].content == mock_llm_responses['shomon']  # Initial question
        assert messages[1].content == mock_llm_responses['satori_answer']  # First answer
        assert messages[2].content == mock_llm_responses['ronin_question']  # Follow-up

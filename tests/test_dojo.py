import pytest
import pytest_asyncio
from unittest.mock import AsyncMock
from openai import AsyncOpenAI
import json
from typing import AsyncGenerator, Optional
from asgiref.sync import sync_to_async
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.django_db]

@pytest.fixture
def mock_llm_responses():
    """Mock responses for various LLM calls."""
    return {
        'dojo_contemplation': {
            'theme': 'Mindful Exploration of Ancient Wisdom',
            'principles': [
                'Embrace uncertainty',
                'Question with respect',
                'Learn through reflection'
            ],
            'ronin_system_message': (
                "You are a seeker of wisdom, walking the path between "
                "knowledge and understanding. Your questions should reflect "
                "deep contemplation and genuine curiosity."
            ),
            'satori_system_message': (
                "You are a guide who illuminates through subtle direction "
                "rather than direct answers. Your responses should encourage "
                "self-discovery and deeper reflection."
            )
        },
        'ronin_meditation': {
            'name': 'Matsuo Basho',
            'interests': ['haiku poetry', 'mountain temples', 'seasonal changes'],
            'style': 'contemplative wandering'
        },
        'satori_meditation': {
            'name': 'Dogen Zenji',
            'specialties': ['zen philosophy', 'mindful living', 'tea ceremony'],
            'teaching_style': 'direct transmission'
        },
        'quest_naming': {
            'quest_title': 'Footprints in Mountain Mist'
        },
        'shomon': "How do the changing seasons reflect the impermanence of our journey?",
        'ronin_question': "What lessons can we learn from the changing autumn leaves?",
        'satori_answer': "As leaves fall, they teach us about letting go with grace. Each descent is a lesson in impermanence.",
        'ronin_reflection': "The leaves don't resist their falling - perhaps there is wisdom in acceptance."
    }

@pytest_asyncio.fixture
async def mock_llm_client(mock_llm_responses):
    """Create a mock LLM client that returns predefined responses."""
    client = AsyncMock(spec=AsyncOpenAI)
    
    # Create nested mock structure to match OpenAI client
    chat_mock = AsyncMock()
    completions_mock = AsyncMock()
    client.chat = chat_mock
    chat_mock.completions = completions_mock
    
    # Track conversation state
    dialogue_check_count = 0
    exchange_count = 0
    message_count = 0
    last_speaker = None
    should_end = False
    
    class MockDelta:
        def __init__(self, content: Optional[str] = None):
            self.content = content

    class MockChoice:
        def __init__(self, delta: MockDelta):
            self.delta = delta

    class MockStreamResponse:
        def __init__(self, content: str):
            self.content = content
            self._chunks = []
            # Split content into smaller chunks for streaming simulation
            chunk_size = 10
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                self._chunks.append(
                    type('Chunk', (), {
                        'choices': [MockChoice(MockDelta(chunk))]
                    })
                )

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._chunks:
                raise StopAsyncIteration
            return self._chunks.pop(0)
    
    async def mock_create(**kwargs):
        nonlocal dialogue_check_count, exchange_count, message_count, last_speaker, should_end
        
        # Get the prompt content
        system_content = kwargs['messages'][0]['content']
        user_content = kwargs['messages'][1]['content'] if len(kwargs['messages']) > 1 else None
        
        # Determine which response to use based on the prompt content
        response_content = None
        
        if "Create a philosophical framework for a Dojo" in system_content:
            response_content = json.dumps(mock_llm_responses['dojo_contemplation'])
        elif "discover your identity as a Ronin" in (user_content or ''):
            response_content = json.dumps(mock_llm_responses['ronin_meditation'])
        elif "reveal your identity as a Satori" in (user_content or ''):
            response_content = json.dumps(mock_llm_responses['satori_meditation'])
        elif "Name this quest" in (user_content or ''):
            response_content = json.dumps(mock_llm_responses['quest_naming'])
        elif "Generate a thoughtful opening question" in system_content:
            response_content = mock_llm_responses['shomon']
            message_count += 1
            last_speaker = 'ronin'
            logger.debug(f"Message {message_count}: Shomon question")
        elif "Respond to this seeker's question" in system_content:
            # Satori's response
            if message_count < 7:  # Only respond if we haven't reached 7 messages
                response_content = mock_llm_responses['satori_answer']
                message_count += 1
                last_speaker = 'satori'
                logger.debug(f"Message {message_count}: Satori answers (exchange {exchange_count})")
                if message_count >= 7:
                    should_end = True
                    logger.debug("Reached 7 messages, setting should_end=True")
            else:
                logger.debug("Would give Satori answer but already at 7 messages")
                should_end = True
        elif "Respond to this guidance" in system_content:
            # Ronin's response
            if message_count < 6:  # We want exactly 7 messages
                response_content = mock_llm_responses['ronin_question']
                message_count += 1
                last_speaker = 'ronin'
                exchange_count += 1
                logger.debug(f"Message {message_count}: Ronin asks question (exchange {exchange_count})")
            else:
                logger.debug("Would give Ronin question but already at 6 messages")
                should_end = True
        elif "You are a dialogue observer" in system_content:
            # After 3 exchanges or 7 messages, end the conversation
            dialogue_check_count += 1
            if should_end or message_count >= 7:
                response_content = json.dumps({
                    'should_end': True,
                    'reason': 'A natural conclusion has been reached'
                })
            else:
                response_content = json.dumps({
                    'should_end': False,
                    'reason': 'The conversation is still ongoing'
                })
            logger.debug(f"Dialogue check {dialogue_check_count}: messages={message_count}, exchanges={exchange_count}, should_end={should_end}")
        
        if response_content is None:
            # Instead of returning "No matching mock response", set should_end=True
            should_end = True
            logger.debug("No mock response available, setting should_end=True")
            response_content = json.dumps({
                'should_end': True,
                'reason': 'A natural conclusion has been reached'
            })
        
        logger.debug(f"Current state - Messages: {message_count}, Exchanges: {exchange_count}, Last speaker: {last_speaker}, Should end: {should_end}")
        
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
    
    completions_mock.create = mock_create
    return client

@pytest_asyncio.fixture
async def real_llm_client():
    """Create a real OpenAI client for integration tests."""
    client = AsyncOpenAI()
    yield client
    await client.close()

@pytest_asyncio.fixture
async def dojo_with_mock_llm(mock_llm_client):
    """Create a Dojo instance with mock LLM client."""
    from dojo.dojo import Dojo
    return Dojo(llm_client=mock_llm_client)

@pytest_asyncio.fixture
async def dojo_with_real_llm(real_llm_client):
    """Create a Dojo instance with real LLM client."""
    from dojo.dojo import Dojo
    dojo = Dojo(llm_client=real_llm_client)
    yield dojo
    await dojo.cleanup()

@pytest_asyncio.fixture
async def initialized_mock_dojo(dojo_with_mock_llm):
    """Create a Dojo instance with initialized cultural context using mocks."""
    dojo = dojo_with_mock_llm
    await dojo.initialize()
    return dojo

@pytest_asyncio.fixture
async def prepared_mock_dojo(initialized_mock_dojo):
    """Create a Dojo instance with prepared Ronin and Satori using mocks."""
    dojo = initialized_mock_dojo
    await dojo.prepare_ronin()
    await dojo.prepare_satori()
    yield dojo
    await dojo.cleanup()

@pytest.fixture
async def mondo_stream() -> AsyncGenerator[str, None]:
    """Simulate an SSE stream of Mondo messages."""
    messages = [
        'data: {"type": "ronin_prepared", "name": "Matsuo Basho"}\n\n',
        'data: {"type": "satori_prepared", "name": "Dogen Zenji"}\n\n',
        'data: {"type": "quest_begun", "title": "Footprints in Mountain Mist"}\n\n',
        'data: {"type": "message", "author": "Matsuo Basho", "content": "How do the changing seasons reflect the impermanence of our journey?"}\n\n'
    ]
    for message in messages:
        yield message

@pytest.mark.mock
async def test_dojo_initialization_mock(dojo_with_mock_llm, mock_llm_responses):
    """Test initializing a Dojo with mock LLM."""
    dojo = dojo_with_mock_llm
    await dojo.initialize()
    
    assert dojo.model.theme == mock_llm_responses['dojo_contemplation']['theme']
    assert dojo.model.principles == mock_llm_responses['dojo_contemplation']['principles']
    assert dojo.model.ronin_system_message == mock_llm_responses['dojo_contemplation']['ronin_system_message']
    assert dojo.model.satori_system_message == mock_llm_responses['dojo_contemplation']['satori_system_message']

@pytest.mark.mock
async def test_dojo_prepare_ronin_mock(initialized_mock_dojo):
    """Test preparing a Ronin with mock LLM."""
    dojo = initialized_mock_dojo
    await dojo.prepare_ronin()
    assert dojo.ronin.name == "Matsuo Basho"
    assert "haiku poetry" in dojo.ronin.interests
    assert dojo.ronin.style == "contemplative wandering"

@pytest.mark.mock
async def test_dojo_prepare_satori_mock(initialized_mock_dojo):
    """Test preparing a Satori with mock LLM."""
    dojo = initialized_mock_dojo
    await dojo.prepare_satori()
    assert dojo.satori.name == "Dogen Zenji"
    assert "zen philosophy" in dojo.satori.specialties

@pytest.mark.mock
async def test_dojo_mondo_mock(prepared_mock_dojo):
    """Test beginning a mondo with mock LLM."""
    dojo = prepared_mock_dojo
    mondo = await dojo.mondo()
    assert mondo.quest.title == "Footprints in Mountain Mist"

@pytest.mark.real
async def test_dojo_initialization_real(dojo_with_real_llm):
    """Test initializing a Dojo with real LLM."""
    dojo = dojo_with_real_llm
    await dojo.initialize()
    
    assert dojo.model.theme
    assert len(dojo.model.principles) > 0
    assert dojo.model.ronin_system_message
    assert dojo.model.satori_system_message

@pytest.mark.real
async def test_dojo_prepare_ronin_real(dojo_with_real_llm):
    """Test preparing a Ronin with real LLM."""
    await dojo_with_real_llm.initialize()
    ronin = await dojo_with_real_llm.prepare_ronin()
    assert hasattr(ronin, 'name')
    assert hasattr(ronin, 'interests')
    assert hasattr(ronin, 'style')

@pytest.mark.mock
async def test_mondo_qa_interaction_mock(prepared_mock_dojo, mock_llm_responses):
    """Test Q&A interaction between Ronin and Satori with mock responses."""
    dojo = prepared_mock_dojo
    
    # Begin the mondo
    mondo = await dojo.mondo()
    
    # Get initial messages
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) == 1  # Just the shomon
    assert messages[0].content == mock_llm_responses['shomon']
    
    # Ronin asks a question
    question = await dojo.ronin.message(mondo, mock_llm_responses['ronin_question'])
    assert question.content == mock_llm_responses['ronin_question']
    assert question.author.name == dojo.ronin.name
    
    # Satori answers
    answer = await dojo.satori.respond(question)
    assert answer.content == mock_llm_responses['satori_answer']
    assert answer.author.name == dojo.satori.name
    
    # Ronin reflects
    reflection = await dojo.ronin.message(mondo, mock_llm_responses['ronin_reflection'])
    assert reflection.content == mock_llm_responses['ronin_reflection']
    assert reflection.author.name == dojo.ronin.name
    
    # Verify final message sequence
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) == 4  # shomon + question + answer + reflection
    assert messages[0].content == mock_llm_responses['shomon']
    assert messages[1].content == mock_llm_responses['ronin_question']
    assert messages[2].content == mock_llm_responses['satori_answer']
    assert messages[3].content == mock_llm_responses['ronin_reflection']

@pytest.mark.real
async def test_mondo_qa_interaction_real(dojo_with_real_llm):
    """Test Q&A interaction between Ronin and Satori with real LLM."""
    # Initialize the dojo first
    await dojo_with_real_llm.initialize()
    
    # Prepare the participants
    await dojo_with_real_llm.prepare_ronin()
    await dojo_with_real_llm.prepare_satori()
    
    # Begin the mondo
    mondo = await dojo_with_real_llm.mondo()
    
    # Ronin asks a question
    question = await dojo_with_real_llm.ronin.message(mondo, "What is the nature of impermanence?")
    assert question.content
    author_name = await sync_to_async(lambda: question.author.name)()
    assert author_name == dojo_with_real_llm.ronin.name
    
    # Satori answers
    answer = await dojo_with_real_llm.satori.respond(question)
    assert answer.content
    author_name = await sync_to_async(lambda: answer.author.name)()
    assert author_name == dojo_with_real_llm.satori.name
    
    # Ronin reflects
    reflection = await dojo_with_real_llm.ronin.message(mondo, "I see now...")
    assert reflection.content
    author_name = await sync_to_async(lambda: reflection.author.name)()
    assert author_name == dojo_with_real_llm.ronin.name
    
    # Verify message sequence
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) == 4  # shomon + question + answer + reflection
    assert all(msg.content for msg in messages)  # All messages should have content
    authors = await sync_to_async(lambda: [msg.author for msg in messages])()
    assert all(authors)   # All messages should have authors

@pytest.mark.real
async def test_mondo_continue_conversation_real(dojo_with_real_llm):
    """Test the automatic conversation continuation between Ronin and Satori."""
    # Initialize the dojo first
    await dojo_with_real_llm.initialize()
    
    # Prepare the participants
    await dojo_with_real_llm.prepare_ronin()
    await dojo_with_real_llm.prepare_satori()
    
    # Begin the mondo
    mondo = await dojo_with_real_llm.mondo()
    
    # Start the conversation loop
    await dojo_with_real_llm.continue_mondo(mondo, max_exchanges=3)
    
    # Verify conversation
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) >= 2  # Should have at least shomon and one response
    
    # Verify alternating speakers
    authors = await sync_to_async(lambda: [msg.author for msg in messages])()
    for i in range(1, len(authors)):
        prev_author = await sync_to_async(lambda: authors[i-1].name)()
        curr_author = await sync_to_async(lambda: authors[i].name)()
        assert prev_author != curr_author, "Speakers should alternate"
    
    # Verify all messages have content
    for msg in messages:
        content = await sync_to_async(lambda: msg.content)()
        assert content, "All messages should have content"

@pytest.mark.mock
async def test_mondo_continue_conversation_mock(prepared_mock_dojo, mock_llm_responses):
    """Test the automatic conversation continuation between Ronin and Satori."""
    dojo = prepared_mock_dojo
    mondo = await dojo.mondo()
    await dojo.continue_mondo(mondo, max_exchanges=2)
    
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) >= 2  # At least shomon + one exchange
    
    # Verify alternating speakers
    authors = await sync_to_async(lambda: [msg.author for msg in messages])()
    for i in range(1, len(authors)):
        prev_author = await sync_to_async(lambda: authors[i-1].name)()
        curr_author = await sync_to_async(lambda: authors[i].name)()
        assert prev_author != curr_author, "Speakers should alternate"

@pytest.mark.mock
async def test_mondo_continue_conversation_custom_length_mock(prepared_mock_dojo):
    """Test conversation continuation with custom max_exchanges."""
    dojo = prepared_mock_dojo
    mondo = await dojo.mondo()
    max_exchanges = 2
    await dojo.continue_mondo(mondo, max_exchanges=max_exchanges)
    
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) <= max_exchanges + 1  # +1 for initial shomon

@pytest.mark.mock
async def test_mondo_continue_conversation_unlimited_mock(prepared_mock_dojo, mock_llm_responses):
    """Test conversation continuation with no exchange limit."""
    dojo = prepared_mock_dojo
    mondo = await dojo.mondo()
    await dojo.continue_mondo(mondo)
    
    messages = [msg async for msg in mondo.messages.all()]
    
    # Log message sequence for debugging
    for i, msg in enumerate(messages):
        content = await sync_to_async(lambda: msg.content)()
        author = await sync_to_async(lambda: msg.author.name)()
        logger.debug(f"Message {i+1}: {author} - {content[:50]}...")
    
    # Should have 7 messages total:
    # 1. Shomon
    # 2-3. First exchange (question + answer)
    # 4-5. Second exchange (question + answer)
    # 6-7. Third exchange (question + answer)
    assert len(messages) == 7, f"Expected 7 messages, got {len(messages)}"
    
    # Verify alternating speakers
    authors = await sync_to_async(lambda: [msg.author for msg in messages])()
    for i in range(1, len(authors)):
        prev_author = await sync_to_async(lambda: authors[i-1].name)()
        curr_author = await sync_to_async(lambda: authors[i].name)()
        assert prev_author != curr_author, f"Speakers should alternate, but got {prev_author} followed by {curr_author}"
    
    # Verify all messages have content
    for msg in messages:
        content = await sync_to_async(lambda: msg.content)()
        assert content, "All messages should have content"
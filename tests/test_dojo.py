import pytest
import pytest_asyncio
from unittest.mock import AsyncMock
from openai import AsyncOpenAI
import json
from typing import AsyncGenerator
from django.db import transaction
from asgiref.sync import sync_to_async

pytestmark = [pytest.mark.django_db]

@pytest.fixture
def mock_llm_responses():
    """Mock responses for various LLM calls."""
    return {
        'ronin_meditation': {
            'name': 'Matsuo Basho',
            'interests': ['haiku poetry', 'mountain temples', 'seasonal changes'],
            'travel_style': 'contemplative wandering'
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
    
    class MockMessage:
        def __init__(self, content: str):
            self.content = content

    class MockResponse:
        def __init__(self, message: MockMessage):
            self.choices = [type('Choice', (), {'message': message})]
    
    async def mock_create(**kwargs):
        # Log the incoming prompt for debugging
        prompt = kwargs['messages'][0]['content']
        print(f"\nIncoming prompt: {prompt}\n")
        
        if "identity as a Ronin" in prompt:
            response_content = json.dumps(mock_llm_responses['ronin_meditation'])
        elif "identity as a Satori" in prompt:
            response_content = json.dumps(mock_llm_responses['satori_meditation'])
        elif "Name this quest" in prompt:
            response_content = json.dumps(mock_llm_responses['quest_naming'])
        elif "opening question" in prompt:
            response_content = mock_llm_responses['shomon']
        elif "ask a question about" in prompt:
            response_content = mock_llm_responses['ronin_question']
        elif "Respond to this seeker's question" in prompt:
            response_content = mock_llm_responses['satori_answer']
        elif "reflect on the answer" in prompt:
            response_content = mock_llm_responses['ronin_reflection']
        elif "You are a dialogue observer" in prompt:
            response_content = json.dumps(mock_llm_responses.get('conclusion_check', {
                'should_end': False,
                'reason': 'The conversation is still ongoing'
            }))
        else:
            print(f"No match found for prompt: {prompt}")
            response_content = "Unknown prompt"
            
        return MockResponse(MockMessage(response_content))
    
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
    from dojo import Dojo
    return Dojo(llm_client=mock_llm_client)

@pytest_asyncio.fixture
async def dojo_with_real_llm(real_llm_client):
    """Create a Dojo instance with real LLM client."""
    from dojo import Dojo
    dojo = Dojo(llm_client=real_llm_client)
    yield dojo
    await dojo.cleanup()

@pytest_asyncio.fixture
async def prepared_mock_dojo(dojo_with_mock_llm):
    """Create a Dojo instance with prepared Ronin and Satori using mocks."""
    dojo = dojo_with_mock_llm
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
async def test_dojo_prepare_ronin_mock(dojo_with_mock_llm):
    """Test preparing a Ronin with mock LLM."""
    dojo = dojo_with_mock_llm
    await dojo.prepare_ronin()
    assert dojo.ronin.name == "Matsuo Basho"
    assert "haiku poetry" in dojo.ronin.interests

@pytest.mark.mock
async def test_dojo_prepare_satori_mock(dojo_with_mock_llm):
    """Test preparing a Satori with mock LLM."""
    dojo = dojo_with_mock_llm
    await dojo.prepare_satori()
    assert dojo.satori.name == "Dogen Zenji"
    assert "zen philosophy" in dojo.satori.specialties

@pytest.mark.mock
async def test_dojo_mondo_mock(prepared_mock_dojo):
    """Test beginning a mondo with mock LLM."""
    dojo = prepared_mock_dojo
    mondo = await dojo.mondo()
    assert mondo.quest.title == "Footprints in Mountain Mist"

@pytest.mark.mock
async def test_dojo_shomon_mock(prepared_mock_dojo):
    """Test generating a shomon with mock LLM."""
    dojo = prepared_mock_dojo
    mondo = await dojo.mondo()
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) == 1
    assert "changing seasons" in messages[0].content.lower()

@pytest.mark.real
async def test_dojo_prepare_ronin_real(dojo_with_real_llm):
    """Test preparing a Ronin with real LLM."""
    ronin = await dojo_with_real_llm.prepare_ronin()
    assert hasattr(ronin, 'name')
    assert hasattr(ronin, 'interests')

@pytest.mark.real
async def test_dojo_prepare_satori_real(dojo_with_real_llm):
    """Test preparing a Satori with real LLM."""
    satori = await dojo_with_real_llm.prepare_satori()
    assert hasattr(satori, 'name')
    assert hasattr(satori, 'specialties')

@pytest.mark.mock
async def test_mondo_qa_interaction_mock(prepared_mock_dojo, mock_llm_responses):
    """Test Q&A interaction between Ronin and Satori with mock responses."""
    dojo = prepared_mock_dojo
    
    # Begin the mondo
    mondo = await dojo.mondo()
    
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
    
    # Verify message sequence
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) == 4  # shomon + question + answer + reflection
    assert messages[0].content == mock_llm_responses['shomon']
    assert messages[1].content == mock_llm_responses['ronin_question']
    assert messages[2].content == mock_llm_responses['satori_answer']
    assert messages[3].content == mock_llm_responses['ronin_reflection']

@pytest.mark.real
async def test_mondo_qa_interaction_real(dojo_with_real_llm):
    """Test Q&A interaction between Ronin and Satori with real LLM."""
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
    """Test the automatic conversation continuation between Ronin and Satori with mocks."""
    # Begin the mondo
    mondo = await prepared_mock_dojo.mondo()
    
    # Add mock response for conclusion check
    mock_llm_responses['conclusion_check'] = {
        'should_end': True,
        'reason': 'A moment of understanding has been reached'
    }
    
    # Start the conversation loop
    await prepared_mock_dojo.continue_mondo(mondo, max_exchanges=3)
    
    # Verify conversation
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) >= 2  # Should have at least shomon and one response
    
    # Verify all messages have content
    for msg in messages:
        content = await sync_to_async(lambda: msg.content)()
        assert content, "All messages should have content"

@pytest.mark.mock
async def test_mondo_continue_conversation_custom_length_mock(prepared_mock_dojo, mock_llm_responses):
    """Test the conversation continuation with custom max_exchanges."""
    # Begin the mondo
    mondo = await prepared_mock_dojo.mondo()
    
    # Set a low max_exchanges value
    max_exchanges = 2
    
    # Start the conversation loop
    await prepared_mock_dojo.continue_mondo(mondo, max_exchanges=max_exchanges)
    
    # Verify conversation
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) <= max_exchanges + 1  # +1 for initial shomon
    
    # Verify all messages have content
    for msg in messages:
        content = await sync_to_async(lambda: msg.content)()
        assert content, "All messages should have content"

@pytest.mark.mock
async def test_mondo_continue_conversation_unlimited_mock(prepared_mock_dojo, mock_llm_responses):
    """Test the conversation continuation with no exchange limit."""
    # Begin the mondo
    mondo = await prepared_mock_dojo.mondo()
    
    # Add mock response for conclusion check that will end after 2 exchanges
    mock_llm_responses['conclusion_check'] = {
        'should_end': False,
        'reason': 'The conversation is still ongoing'
    }
    
    # Start the conversation loop with no exchange limit
    await prepared_mock_dojo.continue_mondo(mondo)
    
    # Verify conversation
    messages = [msg async for msg in mondo.messages.all()]
    assert len(messages) >= 2  # Should have at least shomon and one response
    
    # Verify all messages have content
    for msg in messages:
        content = await sync_to_async(lambda: msg.content)()
        assert content, "All messages should have content"
import pytest
import numpy as np
import re
from unittest.mock import AsyncMock, patch
from openai import AsyncOpenAI
from entities.models import Entity, EntityArchetype, EntityReference
from mondos.models import Mondo, RoninMessage
from dojo.models import Dojo
from quests.models import Quest
from ronins.models import Ronin
from satoris.models import Satori
from satoris.satori import Satori as SatoriService
from asgiref.sync import sync_to_async
import os
import sys

@pytest.fixture
async def test_dojo():
    return await Dojo.objects.acreate(
        theme="Testing Entity Detection",
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
        specialties=["Wisdom", "Understanding"],
        teaching_style="Contemplative",
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
async def test_message(test_mondo, test_ronin):
    return await RoninMessage.objects.acreate(
        mondo=test_mondo,
        content="What is the nature of wisdom and understanding?",
        author=test_ronin
    )

@pytest.fixture
async def openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment")
    return AsyncOpenAI(api_key=api_key)

@pytest.fixture
async def mock_llm_responses():
    return {
        'satori_response': (
            "Wisdom and understanding are interconnected concepts. "
            "Through meditation and mindfulness, we can develop deeper "
            "insights into the nature of reality."
        ),
        'embeddings': np.random.rand(1536).tolist()
    }

class MockStream:
    def __init__(self, content):
        self.content = content
        self.sent = False
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.sent:
            raise StopAsyncIteration
        
        self.sent = True
        return type('Chunk', (), {
            'choices': [type('Choice', (), {
                'delta': type('Delta', (), {
                    'content': self.content
                })
            })]
        })

@pytest.mark.django_db(transaction=True)
@pytest.mark.mock
async def test_satori_entity_detection_mock(test_mondo, test_message, mock_llm_responses):
    """Test entity detection in Satori's responses using mocks."""
    # Clean up existing entities
    await Entity.objects.all().adelete()
    await EntityReference.objects.all().adelete()
    await EntityArchetype.objects.all().adelete()
    
    # Create concept archetype
    concept_archetype = await EntityArchetype.objects.acreate(
        name="concept",
        description="Abstract philosophical concepts",
        embedding=mock_llm_responses['embeddings']
    )
    
    # Create reference entities
    for concept in ["wisdom", "meditation", "mindfulness"]:
        await EntityReference.objects.acreate(
            text=concept,
            archetype=concept_archetype,
            mondo=test_mondo,
            embedding=mock_llm_responses['embeddings']
        )
    
    # Create mock OpenAI client
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = MockStream(mock_llm_responses['satori_response'])
    mock_client.embeddings.create.return_value = type('Response', (), {
        'data': [type('Data', (), {
            'embedding': mock_llm_responses['embeddings']
        })]
    })
    
    # Initialize Satori service
    satori_service = SatoriService(
        name="Test Satori",
        specialties=["Wisdom"],
        teaching_style="Contemplative",
        model_obj=test_message.mondo.quest.satori,
        llm_client=mock_client
    )
    
    # Get Satori's response with entity detection
    response = await satori_service.respond(test_message)
    
    # Verify entities were detected
    entities = await Entity.objects.filter(message=response).acount()
    assert entities > 0, "No entities were detected in Satori's response"
    
    # Verify specific entities
    wisdom_entity = await Entity.objects.select_related('archetype').filter(
        message=response,
        text__iexact="wisdom"
    ).afirst()
    assert wisdom_entity is not None, "Wisdom entity not detected"
    assert await sync_to_async(lambda: wisdom_entity.archetype)() == concept_archetype
    
    meditation_entity = await Entity.objects.select_related('archetype').filter(
        message=response,
        text__iexact="meditation"
    ).afirst()
    assert meditation_entity is not None, "Meditation entity not detected"
    assert await sync_to_async(lambda: meditation_entity.archetype)() == concept_archetype
    
    # Verify entity markup in response
    assert "<entity" in response.content, "No entity markup in response"
    assert "wisdom" in response.content.lower()
    assert "meditation" in response.content.lower()

@pytest.mark.django_db(transaction=True)
@pytest.mark.real
@pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not configured")
async def test_satori_entity_detection_real(test_mondo, test_message, openai_client):
    """Test entity detection in Satori's responses using real OpenAI API."""
    print("\nStarting real API test for entity detection...", file=sys.stderr)
    
    # Clean up existing entities
    await Entity.objects.all().adelete()
    await EntityReference.objects.all().adelete()
    await EntityArchetype.objects.all().adelete()
    
    # Create archetypes with real embeddings
    archetypes = {
        "location": "Physical places, regions, or geographical locations",
        "concept": "Abstract philosophical concepts and ideas",
        "practice": "Spiritual or meditative practices and techniques",
        "person": "Historical or notable figures and teachers"
    }
    
    print("\nCreating archetypes...", file=sys.stderr)
    created_archetypes = {}
    for name, description in archetypes.items():
        print(f"  Creating {name} archetype...", file=sys.stderr)
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=f"{name}: {description}"
        )
        created_archetypes[name] = await EntityArchetype.objects.acreate(
            name=name,
            description=description,
            embedding=embedding_response.data[0].embedding
        )
    
    # Create reference entities for each archetype
    print("\nCreating reference entities...", file=sys.stderr)
    references = {
        "location": ["Mount Fuji", "Zen temple", "meditation hall"],
        "concept": ["mindfulness", "enlightenment", "impermanence"],
        "practice": ["zazen", "meditation", "contemplation"],
        "person": ["Buddha", "Dogen", "Thich Nhat Hanh"]
    }
    
    for archetype_name, entities in references.items():
        archetype = created_archetypes[archetype_name]
        for entity_text in entities:
            print(f"  Creating reference for: {entity_text} ({archetype_name})", file=sys.stderr)
            embedding_response = await openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=entity_text
            )
            await EntityReference.objects.acreate(
                text=entity_text,
                archetype=archetype,
                mondo=test_mondo,
                embedding=embedding_response.data[0].embedding
            )
    
    # Initialize Satori service with verbose output
    class VerboseSatori(SatoriService):
        async def get_system_prompt(self) -> str:
            base_prompt = await super().get_system_prompt()
            return base_prompt + "\n\nPlease incorporate references to specific locations (like Mount Fuji and Zen temples), " + \
                   "meditation practices (like zazen), philosophical concepts (like mindfulness), and teachers (like Dogen) " + \
                   "in your response. Keep the response brief (2-3 sentences) to demonstrate entity detection."
    
    satori_service = VerboseSatori(
        name="Test Satori",
        specialties=["Wisdom", "Zen Practice"],
        teaching_style="Contemplative",
        model_obj=test_message.mondo.quest.satori,
        llm_client=openai_client
    )
    
    # Update test message to encourage entity mentions
    await test_message.adelete()
    test_message = await RoninMessage.objects.acreate(
        mondo=test_mondo,
        content="What is the essence of Zen practice?",
        author=test_message.author
    )
    
    print("\n=== Starting Streaming Response with Entity Detection ===\n", file=sys.stderr)
    response = await satori_service.respond(test_message)
    print("\n=== End of Streaming Response ===\n", file=sys.stderr)
    
    # Get all detected entities
    detected_entities = await Entity.objects.filter(message=response).select_related('archetype').acount()
    print(f"\nTotal entities detected: {detected_entities}", file=sys.stderr)
    
    # Verify we have at least one entity of each type
    for archetype_name, archetype in created_archetypes.items():
        count = await Entity.objects.filter(
            message=response,
            archetype=archetype
        ).acount()
        print(f"\nEntities of type {archetype_name}: {count}", file=sys.stderr)
        assert count > 0, f"No {archetype_name} entities detected"
    
    # Show final response with markup
    print("\nFinal response with entity markup:", file=sys.stderr)
    print(response.content, file=sys.stderr)
    
    # Extract and display entity tags
    print("\nEntity tags found:", file=sys.stderr)
    entity_tags = re.findall(r'<entity[^>]*>.*?</entity>', response.content)
    for tag in entity_tags:
        print(f"  {tag}", file=sys.stderr)
    
    # Basic assertions
    assert "<entity" in response.content, "No entity markup in response"
    assert len(entity_tags) >= 4, "Expected at least 4 different entities" 
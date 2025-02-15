import pytest
from django.conf import settings
import os
import json
import numpy as np
from django.test import Client
from dojo.models import Dojo
from satoris.models import Satori
from quests.models import Quest
from ronins.models import Ronin
from mondos.models import Mondo
from entities.models import EntityArchetype

pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure test environment and settings."""
    # Register test markers
    config.addinivalue_line(
        "markers", "mock: mark test as using mock LLM responses"
    )
    config.addinivalue_line(
        "markers", "real: mark test as using real LLM API"
    )
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "django_db: mark tests that need database access"
    )

def pytest_addoption(parser):
    parser.addoption(
        "--use-real-api",
        action="store_true",
        default=False,
        help="run tests with real API calls"
    )

@pytest.fixture
def use_real_api(request):
    return request.config.getoption("--use-real-api")

@pytest.fixture
def client():
    return Client()

@pytest.fixture
async def test_dojo():
    return await Dojo.objects.acreate(
        theme="Test Dojo",
        principles=["Seek knowledge", "Share wisdom"]
    )

@pytest.fixture
async def test_ronin():
    return await Ronin.objects.acreate(name="Test Ronin")

@pytest.fixture
async def test_satori():
    return await Satori.objects.acreate(name="Test Satori")

@pytest.fixture
async def test_quest(test_dojo, test_ronin, test_satori):
    return await Quest.objects.acreate(
        title="Test Quest",
        dojo=test_dojo,
        ronin=test_ronin,
        satori=test_satori
    )

@pytest.fixture
async def test_mondo(test_quest, test_dojo):
    return await Mondo.objects.acreate(
        quest=test_quest,
        dojo=test_dojo
    )

@pytest.fixture
async def test_archetype():
    return await EntityArchetype.objects.acreate(
        name="Location.City",
        description="A major urban settlement",
        embedding=np.random.rand(1536).tolist()
    )

@pytest.fixture(scope='session')
def django_db_setup(django_db_blocker):
    """Configure test database settings."""
    settings.DATABASES['default'].update({
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'artefact',
        'USER': 'artefact',
        'PASSWORD': 'artefact',
        'HOST': '127.0.0.1',
        'PORT': '5432',
        'ATOMIC_REQUESTS': True,
        'CONN_MAX_AGE': 0,
        'OPTIONS': {
            'client_encoding': 'UTF8',
            'sslmode': 'disable',
        },
        'TEST': {
            'CHARSET': 'UTF8',
            'COLLATION': 'en_US.UTF-8',
            'MIRROR': None,  # Don't mirror the test database
            'SERIALIZE': False,  # Don't serialize database access
            'MIGRATE': True,  # Run migrations for test database
        },
    })
    
    # Log the connection settings for debugging
    print(f"\nTest database connection settings:")
    print(f"Host: {settings.DATABASES['default']['HOST']}")
    print(f"Port: {settings.DATABASES['default']['PORT']} (PostgreSQL 14)")
    print(f"Database: test_{settings.DATABASES['default']['NAME']}")
    print(f"User: {settings.DATABASES['default']['USER']}\n")

@pytest.fixture(autouse=True)
def db_access(db):
    """Give all tests access to the database by default."""
    pass

@pytest.fixture
def mock_llm_responses():
    """Mock responses for LLM calls."""
    return {
        'shomon': "What is the meaning of life?",
        'ronin_question': "How do you find purpose in your daily activities?",
        'satori_answer': "Through mindful engagement with each moment, we discover meaning.",
        'ronin_reflection': "Thank you for sharing your wisdom. I understand better now.",
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
        'archetype_contemplation': [{
            'name': 'concept',
            'description': 'Abstract ideas, principles, or philosophical concepts',
            'examples': ['wisdom', 'enlightenment', 'mindfulness']
        }, {
            'name': 'practice',
            'description': 'Specific methods, techniques, or exercises',
            'examples': ['meditation', 'tea ceremony', 'calligraphy']
        }, {
            'name': 'symbol',
            'description': 'Metaphorical or representative elements',
            'examples': ['mountain', 'path', 'bridge']
        }],
        'ronin_meditation': {
            'name': 'Matsuo Basho',
            'interests': ['haiku poetry', 'mountain temples', 'seasonal changes'],
            'style': 'contemplative wandering',
            'theme': 'Mindful Exploration of Ancient Wisdom',
            'system_prompt': (
                "You are a seeker of wisdom, walking the path between "
                "knowledge and understanding. Your questions should reflect "
                "deep contemplation and genuine curiosity."
            ),
            'meditation_prompt': (
                "In the spirit of mindful exploration, reflect on your role "
                "as a seeker of ancient wisdom."
            )
        },
        'satori_meditation': {
            'name': 'Dogen Zenji',
            'specialties': ['zen philosophy', 'mindful living', 'tea ceremony'],
            'teaching_style': 'direct transmission',
            'theme': 'Mindful Exploration of Ancient Wisdom',
            'system_prompt': (
                "You are a guide who illuminates through subtle direction "
                "rather than direct answers. Your responses should encourage "
                "self-discovery and deeper reflection."
            ),
            'meditation_prompt': (
                "In the spirit of mindful exploration, reflect on your role "
                "as a guide to ancient wisdom."
            )
        },
        'quest_naming': {
            'quest_title': 'Footprints in Mountain Mist'
        },
        'understanding_contemplation': [
            {
                'should_end': False,
                'reason': "I sense there is more to learn about finding purpose through mindfulness."
            },
            {
                'should_end': False,
                'reason': "The connection between acceptance and understanding intrigues me further."
            },
            {
                'should_end': True,
                'reason': "Through our dialogue, I have gained a deeper understanding of how mindful presence reveals life's purpose."
            }
        ]
    }
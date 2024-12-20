import pytest
from django.conf import settings
import os

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
async def async_client():
    from django.test.client import AsyncClient
    return AsyncClient()

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
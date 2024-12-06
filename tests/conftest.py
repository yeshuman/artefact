import pytest
from django.test import AsyncClient
from unittest.mock import AsyncMock
from openai import AsyncOpenAI
from dataclasses import dataclass
import os

@dataclass
class MockDelta:
    content: str = None
    role: str = None

@dataclass
class MockChoice:
    delta: MockDelta
    finish_reason: str = None
    index: int = 0

@dataclass
class MockChunk:
    id: str
    choices: list
    created: int
    model: str
    object: str

def pytest_addoption(parser):
    parser.addoption(
        "--use-real-api",
        action="store_true",
        default=False,
        help="run tests against real OpenAI API"
    )

@pytest.fixture
def use_real_api(request):
    return request.config.getoption("--use-real-api")

@pytest.fixture
async def openai_client(use_real_api):
    """Fixture to provide either a mock or real OpenAI client"""
    if use_real_api:
        # Ensure API key is available for real API calls
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        return AsyncOpenAI()
    
    # Return mock client
    mock_chunks = [
        MockChunk(
            id="1",
            choices=[
                MockChoice(
                    delta=MockDelta(content="Hello")
                )
            ],
            created=1,
            model="gpt-3.5-turbo",
            object="chat.completion.chunk"
        ),
        MockChunk(
            id="2",
            choices=[
                MockChoice(
                    delta=MockDelta(content=" world")
                )
            ],
            created=1,
            model="gpt-3.5-turbo",
            object="chat.completion.chunk"
        )
    ]

    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = mock_chunks
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_stream
    return mock_client

@pytest.fixture
async def async_client():
    return AsyncClient() 
import pytest
from django.urls import reverse
from unittest.mock import AsyncMock, patch
from django.test import AsyncClient
import asyncio

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_chat_stream_response_headers(async_client):
    """Test that chat stream response has correct SSE headers"""
    url = reverse('chat_stream')
    
    response = await async_client.get(url)
    
    assert response.status_code == 200
    assert response['Content-Type'] == 'text/event-stream'
    assert response['Cache-Control'] == 'no-cache'
    assert response['X-Accel-Buffering'] == 'no'

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_chat_stream_content(async_client, openai_client):
    """Test that chat stream properly streams AI responses"""
    url = reverse('chat_stream')

    with patch('chats.views.AsyncOpenAI', return_value=openai_client):
        # Create a message to trigger streaming
        from chats.models import HumanMessage
        response = await async_client.post(reverse('chat_post'), {'message': 'Test message'})
        assert response.status_code == 200
        
        # Get streaming response
        response = await async_client.get(url)
        
        # Read the response content with timeout
        response_content = []
        chunk_count = 0
        max_chunks = 10  # Limit number of chunks to reads
        try:
            async for chunk in response.streaming_content:
                response_content.append(chunk.decode())
                chunk_count += 1
                if chunk_count >= max_chunks:
                    break
                # Add a small delay to prevent busy-waiting
                await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            pytest.fail("Streaming response timed out")

        # Verify the streamed content
        if isinstance(openai_client, AsyncMock):
            # For mock client, verify exact responses
            assert any('Hello' in chunk for chunk in response_content)
            assert any('world' in chunk for chunk in response_content)
        else:
            # For real API, verify structure only
            assert any('event: ai' in chunk for chunk in response_content)
            assert any('data: <span class=\'chunk\'>' in chunk for chunk in response_content)
        
        # Common assertions
        assert all('event: ai' in chunk for chunk in response_content if '<span class=\'chunk\'>' in chunk)

from django.db import models
from asgiref.sync import sync_to_async
from entities.models import EntityArchetype
from mondos.models import Message
from entities.services import StreamingEntityDetector
from dojo.dojo import Sensei
from typing import Optional, Any, TYPE_CHECKING, Dict, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mondos.models import Mondo, Message
    from openai import AsyncOpenAI

class Satori(Sensei):
    """A guide in the dojo who responds to the Ronin's questions."""
    
    def __init__(self, name: str, model_instance: 'Satori', llm_client: 'AsyncOpenAI'):
        """Initialize a Satori instance.
        
        Args:
            name: The name of this Satori instance
            model_instance: The database model instance for this Satori
            llm_client: The OpenAI client for LLM interactions
        """
        self.name = name
        self.model_instance = model_instance
        self.llm_client = llm_client

    async def respond(self, mondo, message):
        """Generate a response to the Ronin's message."""
        
        # Get archetypes for entity detection
        archetypes = await sync_to_async(list)(EntityArchetype.objects.all())
        
        # Initialize conversation context
        conversation = [
            {
                "role": "system",
                "content": self.model_instance.system_prompt
            }
        ]

        # Add message history
        async for msg in Message.objects.filter(mondo=mondo).order_by('created_at'):
            conversation.append({
                "role": "user" if msg.author == "ronin" else "assistant",
                "content": msg.content
            })

        # Get response from LLM
        response = await self.llm_client.chat.completions.create(
            model=self.model_instance.model_name,
            messages=conversation,
            temperature=0.7,
            stream=True
        )

        # Process response stream and detect entities
        full_response = ""
        detectors = {
            archetype.name: StreamingEntityDetector(message_id=message.id)
            for archetype in archetypes
        }
        for detector in detectors.values():
            detector.reset()

        async for chunk in response:
            if not chunk.choices[0].delta.content:
                continue
                
            text_chunk = chunk.choices[0].delta.content
            full_response += text_chunk

            # Detect entities in each archetype
            for archetype in archetypes:
                detector = detectors[archetype.name]
                detector.archetype = archetype
                marked_text, entities = await detector.process_chunk(
                    text_chunk,
                    message.id,
                    self.llm_client.embeddings.create
                )
                text_chunk = marked_text

        # Update message with marked text
        message.content = full_response
        await sync_to_async(message.save)()

        return message

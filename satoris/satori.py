from typing import Optional, Any, TYPE_CHECKING, Dict, List
from asgiref.sync import sync_to_async
from dojo.dojo import Sensei
from .models import Satori  # DB Model class
from mondos.models import SatoriMessage  # For type checking in respond method
from entities.services import StreamingEntityDetector
from entities.models import EntityArchetype
import numpy as np
import logging
import sys
import re

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mondos.models import Mondo, Message, SatoriMessage


class Satori(Sensei):
    """A guide who illuminates the path to understanding.
    
    The Satori's role is to guide and illuminate, helping seekers
    discover deeper understanding through dialogue and reflection.
    """
    
    def __init__(
        self,
        name: str,
        specialties: list[str],
        teaching_style: str,
        model_obj: Optional[Satori] = None,  # DB record
        llm_client: Optional[Any] = None
    ):
        """Initialize a new Satori instance.
        
        Args:
            name: The Satori's given name
            specialties: Areas of expertise
            teaching_style: Natural approach to guiding others
            model_obj: Optional Django model instance
            llm_client: Optional LLM client for interactions
        """
        super().__init__(name, model_obj, llm_client)
        self.specialties = specialties
        self.teaching_style = teaching_style
        self.model_obj = model_obj  # Store reference to DB record
    
    async def message(self, mondo: 'Mondo', content: str) -> 'SatoriMessage':
        """Share wisdom or guidance in a Mondo.
        
        Args:
            mondo: The dialogue context
            content: The guidance or wisdom to share
            
        Returns:
            Message: The created SatoriMessage
        """
        from mondos.models import SatoriMessage
        return await SatoriMessage.objects.acreate(
            mondo=mondo,
            content=content,
            author=self.model_obj
        )
    
    async def get_system_prompt(self) -> str:
        """Get the system prompt for this Satori.
        
        Returns:
            str: The system prompt for LLM interactions
        """
        if self.model_obj and self.model_obj.system_prompt:
            # Use stored dynamic prompt with interpolated values
            return self.model_obj.system_prompt.format(
                name=self.name,
                specialties=', '.join(self.specialties),
                teaching_style=self.teaching_style
            )
        
        # Fallback to default prompt
        return (
            f"You are a guide named {self.name} with expertise in "
            f"{', '.join(self.specialties)} and a {self.teaching_style} "
            f"approach to teaching. Your role is to illuminate understanding "
            f"through thoughtful dialogue.\n\n"
            f"Consider the full context of your conversation so far and "
            f"respond with wisdom that guides your seeker toward deeper insights."
        )
    
    async def get_meditation_prompt(self) -> str:
        """Get the meditation prompt for Satori self-discovery."""
        if self.model_obj and self.model_obj.meditation_prompt:
            return self.model_obj.meditation_prompt
            
        # Fallback to default prompt
        return (
            "Through deep meditation, discover your identity as a guide:\n"
            "1. Your name (a meaningful name that fits the dojo's theme)\n"
            "2. Your areas of expertise and wisdom\n"
            "3. Your teaching style (if not already specified)\n\n"
            "Consider the dojo's theme and cultural context for inspiration.\n"
            "Respond in JSON format with keys: name (string), specialties (list), and teaching_style (string)"
        )
    
    async def create_from_meditation(self, meditation_data: Dict[str, Any]) -> None:
        """Create or update Satori from meditation data."""
        # Update instance attributes
        self.name = meditation_data['name']
        self.specialties = meditation_data['specialties']
        if not self.teaching_style:  # Only update if not provided during initialization
            self.teaching_style = meditation_data['teaching_style']
        
        # Create or update DB record
        from satoris.models import Satori as SatoriModel
        if self.model_obj is None:
            self.model_obj = await SatoriModel.objects.acreate(
                name=self.name,
                specialties=self.specialties,
                teaching_style=self.teaching_style
            )
        else:
            self.model_obj.name = self.name
            self.model_obj.specialties = self.specialties
            self.model_obj.teaching_style = self.teaching_style
            await sync_to_async(self.model_obj.save)()
    
    async def respond(self, message: 'Message') -> 'SatoriMessage':
        """Generate a response to the given message with entity detection.
        
        Args:
            message: The message to respond to
            
        Returns:
            SatoriMessage: The response message with entity markup
        """
        # Get mondo context
        mondo = message.mondo
        
        # Get system prompt
        system_prompt = await self.get_system_prompt()
        
        # Prepare conversation history
        messages = [{
            "role": "system",
            "content": system_prompt
        }]
        
        # Add previous messages for context
        async for msg in message.mondo.messages.order_by('created_at'):
            # Get message type safely
            msg_type = await sync_to_async(lambda m=msg: m.__class__.__name__)()
            role = "assistant" if msg_type == "SatoriMessage" else "user"
            
            messages.append({
                "role": role,
                "content": await sync_to_async(lambda m=msg: m.content)()
            })
            
        # Get streaming response from LLM
        print("\nProcessing Satori response...", file=sys.stderr)
        print(f"\nRonin's question: {await sync_to_async(lambda m=message: m.content)()}", file=sys.stderr)
        print("\nGenerating response from LLM...", file=sys.stderr)
        
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            stream=True
        )
        
        # Initialize entity detector and get archetypes
        detector = StreamingEntityDetector(mondo_id=mondo.id)
        archetypes = await sync_to_async(list)(EntityArchetype.objects.all())
        print(f"\nFound {len(archetypes)} archetypes to check", file=sys.stderr)

        # Create embedding function using OpenAI with context
        async def get_embedding_with_context(text: str) -> np.ndarray:
            # Get context from detector's buffer
            context = detector._get_context()
            # Include surrounding context if available
            input_text = f"{context} {text} {context}".strip() if context else text
            response = await self.llm_client.embeddings.create(
                model="text-embedding-ada-002",
                input=input_text
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        # Create the message first with empty content
        response = await self.message(mondo, "")
        
        # Process the stream in real-time
        full_response = []
        buffer = []  # Buffer for word completion
        print("\nStreaming response with real-time entity detection:", file=sys.stderr)
        
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    print("\nReceived chunk:", content_chunk, file=sys.stderr)
                    
                    # Add to buffer and check if we have a complete word
                    buffer.append(content_chunk)
                    current_text = ''.join(buffer)
                    
                    # Process if we have a complete word (ends with space or punctuation)
                    if current_text.strip() and (content_chunk[-1].isspace() or content_chunk[-1] in '.,!?;:'):
                        marked_chunk = current_text
                        
                        # Track best archetype matches
                        entity_matches = {}  # text -> {archetype, confidence, markup}
                        
                        # Process with each archetype
                        for archetype in archetypes:
                            detector.archetype = archetype
                            marked_text, entities = await detector.process_chunk(
                                marked_chunk,
                                response.id,
                                get_embedding_with_context
                            )
                            
                            if entities:
                                print(f"\nDetected entities with {await sync_to_async(lambda a=archetype: a.name)()}:", file=sys.stderr)
                                for entity in entities:
                                    print(f"  - {entity['text']} (confidence: {entity['confidence']:.2f})", file=sys.stderr)
                                    
                                    # Track best archetype match
                                    if entity['text'] not in entity_matches or entity['confidence'] > entity_matches[entity['text']]['confidence']:
                                        entity_matches[entity['text']] = {
                                            'archetype': archetype,
                                            'confidence': entity['confidence'],
                                            'markup': f'<entity id="{entity["id"]}" type="{await sync_to_async(lambda a=archetype: a.name)()}">{entity["text"]}</entity>'
                                        }
                        
                        # Apply best archetype matches
                        final_chunk = marked_chunk
                        for entity_text, match in entity_matches.items():
                            pattern = re.compile(rf'\b{re.escape(entity_text)}\b', re.IGNORECASE)
                            final_chunk = pattern.sub(match['markup'], final_chunk)
                        
                        print("\nProcessed chunk with markup:", final_chunk, file=sys.stderr)
                        full_response.append(final_chunk)
                        response.content = "".join(full_response)
                        await response.asave()
                        buffer = []  # Clear buffer
                    else:
                        # Keep collecting chunks if we don't have a complete word
                        continue

        if buffer:
            remaining_text = ''.join(buffer)
            if remaining_text.strip():
                full_response.append(remaining_text)
                response.content = "".join(full_response)
                await response.asave()
        
        return response

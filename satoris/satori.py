from typing import Optional, Any, TYPE_CHECKING, Dict, List
from asgiref.sync import sync_to_async
from dojo.dojo import Sensei
from .models import Satori  # DB Model class
from mondos.models import SatoriMessage  # For type checking in respond method
from entities.services import StreamingEntityDetector
from entities.models import EntityArchetype
import numpy as np
import logging

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
        """Respond to a seeker's question with guidance.
        
        Args:
            message: The message to respond to
            
        Returns:
            Message: The guidance response
        """
        if not self.llm_client:
            raise ValueError("Satori requires an LLM client to formulate responses")
        
        # Get message content and mondo safely
        content = await sync_to_async(lambda: message.content)()
        mondo = await sync_to_async(lambda: message.mondo)()
        
        # Get conversation history
        messages = [msg async for msg in mondo.messages.all()]
        history = []
        for msg in messages:
            msg_content = await sync_to_async(lambda: msg.content)()
            msg_author = await sync_to_async(lambda: msg.author.name)()
            role = "assistant" if isinstance(msg, SatoriMessage) else "user"
            history.append({"role": role, "content": msg_content})
            
        # Use LLM to generate guidance with conversation history
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": await self.get_system_prompt()
            }] + history,
            stream=True
        )
        
        # Collect the full response first
        full_response = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    logger.info(f"Satori response chunk: {content_chunk}")
                    full_response.append(content_chunk)
        
        response_content = ''.join(full_response)
        
        # Create the message first
        response = await self.message(mondo, response_content)
        
        # Initialize entity detector
        detector = StreamingEntityDetector(mondo_id=mondo.id)
        
        # Get all archetypes for entity detection
        archetypes = await sync_to_async(list)(EntityArchetype.objects.all())
        
        # Create embedding function using OpenAI
        async def get_embedding(text: str) -> np.ndarray:
            response = await self.llm_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        
        # Process the full response with each archetype
        marked_content = response_content
        for archetype in archetypes:
            detector.archetype = archetype
            marked_text, entities = await detector.process_chunk(
                response_content,
                response.id,  # Use the created message's ID
                get_embedding
            )
            if entities:
                marked_content = marked_text
                logger.info(f"Detected entities: {entities}")
        
        # Update the message with marked-up content
        response.content = marked_content
        await sync_to_async(response.save)()
        
        return response

from typing import Optional, Any, TYPE_CHECKING
from asgiref.sync import sync_to_async
from dojo.dojo import Sensei
from .models import Satori  # DB Model class
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mondos.models import Mondo, Message, SatoriMessage


class Satori(Sensei):
    """A guide who helps illuminate the path to understanding artifacts."""
    
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
            specialties: Areas of deep understanding
            teaching_style: Approach to guiding others
            model_obj: Optional Django model instance
            llm_client: Optional LLM client for interactions
        """
        super().__init__(name, model_obj, llm_client)
        self.specialties = specialties
        self.teaching_style = teaching_style
        self.model_obj = model_obj  # Store reference to DB record
    
    async def message(self, mondo: 'Mondo', content: str) -> 'SatoriMessage':
        """Share guidance or insight in a Mondo.
        
        Args:
            mondo: The dialogue context
            content: The wisdom or guidance to share
            
        Returns:
            Message: The created SatoriMessage
        """
        from mondos.models import SatoriMessage
        return await SatoriMessage.objects.acreate(
            mondo=mondo,
            content=content,
            author=self.model_obj
        )
    
    async def respond(self, message: 'Message') -> 'SatoriMessage':
        """Respond to a Ronin's question with guidance.
        
        Args:
            message: The message to respond to
            
        Returns:
            Message: The guidance offered
        """
        if not self.llm_client:
            raise ValueError("Satori requires an LLM client to formulate responses")
        
        # Get message content safely
        content = await sync_to_async(lambda: message.content)()
        mondo = await sync_to_async(lambda: message.mondo)()
            
        # Use LLM to generate guidance
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": (
                    f"You are a Satori named {self.name} with specialties in "
                    f"{', '.join(self.specialties)} and a {self.teaching_style} "
                    f"teaching style.\n\n"
                    f"Respond to this seeker's question with wisdom and guidance. "
                    f"Naturally mention relevant places, cultural sites, or historical "
                    f"locations that illustrate your teachings."
                )
            }, {
                "role": "user",
                "content": content
            }],
            stream=True
        )
        
        # Collect streamed response
        full_response = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    logger.info(f"Satori response chunk: {content_chunk}")
                    full_response.append(content_chunk)
        
        response_content = ''.join(full_response)
        return await self.message(mondo, response_content)

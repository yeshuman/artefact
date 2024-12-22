from typing import Optional, Any, TYPE_CHECKING
from asgiref.sync import sync_to_async
from dojo.dojo import Sensei
from .models import Ronin  # DB Model class
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mondos.models import Mondo, Message, RoninMessage


class Ronin(Sensei):
    """A wandering seeker in search of the world's artifacts and wisdom.
    
    The Ronin's journey is one of discovery and questioning. They observe
    the world through the lens of their interests and personal style, forming
    questions that lead to deeper understanding.
    """
    
    def __init__(
        self,
        name: str,
        interests: list[str],
        style: str,
        model_obj: Optional[Ronin] = None,  # DB record
        llm_client: Optional[Any] = None
    ):
        """Initialize a new Ronin instance.
        
        Args:
            name: The Ronin's given name
            interests: List of interests and areas of exploration
            style: Personal approach and characteristics
            model_obj: Optional Django model instance
            llm_client: Optional LLM client for interactions
        """
        super().__init__(name, model_obj, llm_client)
        self.interests = interests
        self.style = style
        self.model_obj = model_obj  # Store reference to DB record
    
    async def message(self, mondo: 'Mondo', content: str) -> 'RoninMessage':
        """Create a question or reflection in a Mondo.
        
        Args:
            mondo: The dialogue context
            content: The question or reflection
            
        Returns:
            Message: The created RoninMessage
        """
        from mondos.models import RoninMessage
        return await RoninMessage.objects.acreate(
            mondo=mondo,
            content=content,
            author=self.model_obj
        )
    
    async def respond(self, message: 'Message') -> 'RoninMessage':
        """Respond to a Satori's guidance with further questions.
        
        Args:
            message: The message to respond to
            
        Returns:
            Message: The follow-up question
        """
        if not self.llm_client:
            raise ValueError("Ronin requires an LLM client to formulate responses")
        
        # Get message content safely
        content = await sync_to_async(lambda: message.content)()
        mondo = await sync_to_async(lambda: message.mondo)()
            
        # Use LLM to generate a follow-up question
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": (
                    f"You are a Ronin named {self.name} with interests in "
                    f"{', '.join(self.interests)} and a {self.style} "
                    f"approach. You are on a journey of discovery.\n\n"
                    f"Consider the guidance you've received and respond with "
                    f"thoughtful questions or reflections that deepen your understanding."
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
                    logger.info(f"Ronin response chunk: {content_chunk}")
                    full_response.append(content_chunk)
        
        response_content = ''.join(full_response)
        return await self.message(mondo, response_content)

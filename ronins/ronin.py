from typing import Optional, Any, TYPE_CHECKING
from dojo import Sensei
from .models import Ronin  # DB Model class

if TYPE_CHECKING:
    from mondos.models import Mondo, Message, RoninMessage


class Ronin(Sensei):
    """A wandering seeker in search of the world's artifacts and wisdom.
    
    The Ronin's journey is one of discovery and questioning. They observe
    the world through the lens of their interests and travel style, forming
    questions that lead to deeper understanding.
    """
    
    def __init__(
        self,
        name: str,
        interests: list[str],
        travel_style: str,
        model_obj: Optional[Ronin] = None,  # DB record
        llm_client: Optional[Any] = None
    ):
        """Initialize a new Ronin instance.
        
        Args:
            name: The Ronin's given name
            interests: List of travel-related interests
            travel_style: Preferred style of travel
            model_obj: Optional Django model instance
            llm_client: Optional LLM client for interactions
        """
        super().__init__(name, model_obj, llm_client)
        self.interests = interests
        self.travel_style = travel_style
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
            
        # Use LLM to generate a follow-up question
        response = await self.llm_client.complete(
            messages=[{"role": "user", "content": message.content}]
        )
        
        return await self.message(message.mondo, response)

from typing import Optional, Any, TYPE_CHECKING
from dojo import Sensei
from .models import Satori as SatoriModel

if TYPE_CHECKING:
    from mondos.models import Mondo, Message, SatoriMessage


class Satori(Sensei):
    """A guide who helps illuminate the path to understanding artifacts."""
    
    def __init__(
        self,
        name: str,
        specialties: list[str],
        teaching_style: str,
        model_instance: Optional[SatoriModel] = None,
        llm_client: Optional[Any] = None
    ):
        """Initialize a new Satori instance.
        
        Args:
            name: The Satori's given name
            specialties: Areas of deep understanding
            teaching_style: Approach to guiding others
            model_instance: Optional Django model instance
            llm_client: Optional LLM client for interactions
        """
        super().__init__(name, model_instance, llm_client)
        self.specialties = specialties
        self.teaching_style = teaching_style
    
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
            content=content
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
            
        # Use LLM to generate guidance
        response = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": (
                    f"You are a Satori named {self.name} with specialties in "
                    f"{', '.join(self.specialties)} and a {self.teaching_style} "
                    f"teaching style. Respond to this seeker's question with wisdom and guidance."
                )
            }, {
                "role": "user",
                "content": message.content
            }]
        )
        
        return await self.message(message.mondo, response.choices[0].message.content)

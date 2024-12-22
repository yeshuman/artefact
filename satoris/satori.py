from typing import Optional, Any, TYPE_CHECKING, Dict
from asgiref.sync import sync_to_async
from dojo.dojo import Sensei
from .models import Satori  # DB Model class
from mondos.models import SatoriMessage  # For type checking in respond method
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
                "content": (
                    f"You are a Satori named {self.name} with specialties in "
                    f"{', '.join(self.specialties)} and a {self.teaching_style} "
                    f"teaching style.\n\n"
                    f"Respond to this seeker's question with wisdom and guidance. "
                    f"Naturally mention relevant places, cultural sites, or historical "
                    f"locations that illustrate your teachings. "
                    f"Consider the full context of your conversation so far."
                )
            }] + history,
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
    
    async def get_system_prompt(self) -> str:
        """Get the system prompt for this Satori.
        
        Returns:
            str: The system prompt for LLM interactions
        """
        return (
            f"You are a Satori named {self.name} with specialties in "
            f"{', '.join(self.specialties)} and a {self.teaching_style} "
            f"teaching style.\n\n"
            f"Respond to this seeker's question with wisdom and guidance. "
            f"Naturally mention relevant places, cultural sites, or historical "
            f"locations that illustrate your teachings. "
            f"Consider the full context of your conversation so far."
        )
    
    async def get_meditation_prompt(self) -> str:
        """Get the meditation prompt for Satori self-discovery."""
        return (
            "Through profound meditation, reveal your identity as a Satori:\n"
            "1. Your name (a meaningful Japanese name)\n"
            "2. Your three areas of specialty and deep understanding\n"
            "3. Your natural approach to guiding others (if not already specified)\n\n"
            "Consider historical Zen masters, teachers, and philosophers for inspiration.\n"
            "Respond in JSON format with keys: name (string), specialties (list), and teaching_style (string)"
        )
    
    async def create_from_meditation(self, meditation_data: Dict[str, Any]) -> None:
        """Create or update Satori from meditation data."""
        # Handle British English spelling if present
        if 'specialities' in meditation_data and 'specialties' not in meditation_data:
            logger.info("Converting British spelling 'specialities' to American 'specialties'")
            meditation_data['specialties'] = meditation_data.pop('specialities')
        
        # Update instance attributes
        self.name = meditation_data['name']
        self.specialties = meditation_data['specialties']
        if not self.teaching_style:  # Only update if not provided during initialization
            self.teaching_style = meditation_data['teaching_style']
        
        # Create or update DB record
        from satoris.models import Satori as SatoriModel
        if self.model is None:
            self.model = await SatoriModel.objects.acreate(
                name=self.name,
                specialties=self.specialties,
                teaching_style=self.teaching_style
            )
            self.model_obj = self.model  # Also set model_obj for compatibility

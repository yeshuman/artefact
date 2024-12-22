from typing import Optional, Any, TYPE_CHECKING, Dict
from asgiref.sync import sync_to_async
from dojo.dojo import Sensei
from .models import Ronin  # DB Model class
from mondos.models import SatoriMessage  # For type checking in respond method
import logging
import json

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
            
        # Use LLM to generate a follow-up question with conversation history
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": (
                    f"You are a Ronin named {self.name} with interests in "
                    f"{', '.join(self.interests)} and a {self.style} "
                    f"approach. You are on a journey of discovery.\n\n"
                    f"Consider the full context of your conversation so far and "
                    f"respond with thoughtful questions or reflections that deepen "
                    f"your understanding."
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
                    logger.info(f"Ronin response chunk: {content_chunk}")
                    full_response.append(content_chunk)
        
        response_content = ''.join(full_response)
        return await self.message(mondo, response_content)
    
    async def get_system_prompt(self) -> str:
        """Get the system prompt for this Ronin.
        
        Returns:
            str: The system prompt for LLM interactions
        """
        return (
            f"You are a Ronin named {self.name} with interests in "
            f"{', '.join(self.interests)} and a {self.style} "
            f"approach. You are on a journey of discovery.\n\n"
            f"Consider the full context of your conversation so far and "
            f"respond with thoughtful questions or reflections that deepen "
            f"your understanding."
        )
    
    async def get_meditation_prompt(self) -> str:
        """Get the meditation prompt for Ronin self-discovery."""
        return (
            "Through deep meditation, discover your identity as a Ronin:\n"
            "1. Your name (a meaningful Japanese name)\n"
            "2. Your three main interests in exploring the world\n"
            "3. Your personal style and approach (if not already specified)\n\n"
            "Consider historical wandering monks, scholars, and artists for inspiration.\n"
            "Respond in JSON format with keys: name (string), interests (list), and style (string)"
        )
    
    async def create_from_meditation(self, meditation_data: Dict[str, Any]) -> None:
        """Create or update Ronin from meditation data."""
        # Update instance attributes
        self.name = meditation_data['name']
        self.interests = meditation_data['interests']
        if not self.style:  # Only update if not provided during initialization
            self.style = meditation_data['style']
        
        # Create or update DB record
        from ronins.models import Ronin as RoninModel
        if self.model_obj is None:
            self.model_obj = await RoninModel.objects.acreate(
                name=self.name,
                interests=self.interests,
                style=self.style
            )

    async def contemplate_quest(self) -> str:
        """Contemplate and name a quest that aligns with interests and style.
        
        Returns:
            str: The title of the contemplated quest
        """
        # Stream quest contemplation
        quest_response = await self.stream_llm_response([{
            "role": "system",
            "content": (
                f"You are a Ronin named {self.name} with interests in "
                f"{', '.join(self.interests)} and a {self.style} "
                "approach.\n\n"
                "Through meditation, envision the quest you wish to undertake. "
                "What profound question or exploration calls to you?\n\n"
                "Name this quest in a way that reflects its depth and your seeking nature. "
                "Consider the style of titles like 'In Search of Lost Wisdom' or 'The Path Through Ancient Gardens'.\n\n"
                "Respond in JSON format with key: quest_title (string)"
            )
        }])
        
        # Clean and parse the response
        try:
            cleaned_json = self._clean_json_response(quest_response)
            contemplation = json.loads(cleaned_json)
            logger.info(f"Quest contemplation: {contemplation['quest_title']}")
            return contemplation['quest_title']
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse quest contemplation: {quest_response}")
            raise

    async def contemplate_understanding(self, message_content: str) -> Dict[str, Any]:
        """Contemplate whether you've reached understanding or should continue seeking.
        
        As a Ronin, reflect on the latest exchange and determine if you've reached
        a satisfactory level of understanding for your current quest, or if there
        are still depths to explore.
        
        Args:
            message_content: The content of the latest message to contemplate
            
        Returns:
            Dict with keys:
                - should_end (bool): Whether to conclude the dialogue
                - reason (str): Reflection on why to continue or conclude
        """
        # Stream contemplation response
        contemplation = await self.stream_llm_response([{
            "role": "system",
            "content": (
                f"You are {self.name}, a seeker of wisdom with interests in "
                f"{', '.join(self.interests)} and a {self.style} approach.\n\n"
                "Consider the latest exchange in your dialogue. Have you reached a state "
                "of understanding that satisfies your quest? Or do you have more to explore?\n\n"
                "Respond in JSON format with keys:\n"
                "- should_end (boolean): true if you've reached understanding\n"
                "- reason (string): your reflection on why you choose to continue or conclude"
            )
        }, {
            "role": "user",
            "content": message_content
        }])
        
        # Clean and parse the response
        try:
            cleaned_json = self._clean_json_response(contemplation)
            decision = json.loads(cleaned_json)
            logger.info(f"Understanding contemplation: {json.dumps(decision, indent=2)}")
            return decision
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse understanding contemplation: {contemplation}")
            raise

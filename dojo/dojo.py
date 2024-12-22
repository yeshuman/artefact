from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from openai import AsyncOpenAI
from asgiref.sync import sync_to_async
import json
import logging
import asyncio
from dojo.models import Dojo as DojoModel
from quests.models import Quest
from mondos.models import Mondo, RoninMessage, SatoriMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dojo.models import Dojo as DojoModel
    from mondos.models import Mondo, Message, RoninMessage, SatoriMessage
    from ronins.models import Ronin as RoninModel
    from satoris.models import Satori as SatoriModel
    from ronins.ronin import Ronin
    from satoris.satori import Satori


class Sensei:
    """Base class for enlightened beings in our simulation.
    
    A Sensei (先生) - "one who comes before" - represents the common essence
    of both seekers and guides in our system. Whether discovering or sharing
    wisdom, all inherit the basic abilities to observe, remember, and contemplate.
    
    Attributes:
        name: The being's given name
        model: Associated Django model instance if any
        llm_client: Client for LLM interactions
    """
    
    def __init__(
        self,
        name: str,
        model_instance: Optional[Any] = None,
        llm_client: Optional[Any] = None
    ):
        """Initialize a new Sensei instance.
        
        Args:
            name: The being's given name
            model_instance: Optional Django model instance
            memory_limit: Maximum memories to retain (default: 100)
            llm_client: Optional LLM client for interactions
        """
        self.name = name
        self.model = model_instance
        self.llm_client = llm_client
    
    async def message(self, mondo: 'Mondo', content: str) -> 'Message':
        """Create and emit a message in a Mondo.
        
        Args:
            mondo: The dialogue context
            content: The message content
            
        Returns:
            Message: The created message
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Each being must define their own way of messaging")
    
    async def get_system_prompt(self) -> str:
        """Get the system prompt for this being.
        
        Returns:
            str: The system prompt for LLM interactions
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Each being must define their own system prompt")
    
    async def get_meditation_prompt(self) -> str:
        """Get the meditation prompt for self-discovery.
        
        Returns:
            str: The meditation prompt for discovering identity
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Each being must define their own meditation prompt")
    
    async def create_from_meditation(self, meditation_data: Dict[str, Any]) -> None:
        """Create or update self from meditation data.
        
        Args:
            meditation_data: The parsed meditation response
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Each being must define how to create from meditation")
    
    async def prepare_self(self, system_message: str) -> None:
        """Discover identity through meditation.
        
        Args:
            system_message: The system message guiding meditation
        """
        if not self.llm_client:
            raise ValueError(f"{self.__class__.__name__} requires an LLM client for meditation")
        
        # Get meditation prompt
        meditation_prompt = await self.get_meditation_prompt()
        
        # Stream meditation response
        meditation_response = await self.stream_llm_response([{
            "role": "system",
            "content": system_message
        }, {
            "role": "user",
            "content": meditation_prompt
        }])
        
        # Clean and parse the response
        try:
            cleaned_json = self._clean_json_response(meditation_response)
            meditation_data = json.loads(cleaned_json)
            logger.info(f"Parsed meditation data:\n{json.dumps(meditation_data, indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse meditation response: {meditation_response}")
            raise
        
        # Create or update self from meditation
        await self.create_from_meditation(meditation_data)
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response that might be wrapped in markdown."""
        if response.startswith('```') and response.endswith('```'):
            lines = response.split('\n')
            lines = lines[1:-1]
            if lines[0].lower() in ['json', 'javascript']:
                lines = lines[1:]
            response = '\n'.join(lines)
        return response.strip()
    
    async def stream_llm_response(self, messages: List[Dict[str, str]], model: str = "gpt-4-1106-preview") -> str:
        """Stream a response from the LLM and collect the result.
        
        Args:
            messages: List of message dictionaries for the LLM
            model: The model to use for generation
            
        Returns:
            str: The complete response from the LLM
        """
        if not self.llm_client:
            raise ValueError(f"{self.__class__.__name__} requires an LLM client to formulate responses")
        
        stream = await self.llm_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        # Collect streamed response
        response_chunks = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    logger.info(f"{self.name} response chunk: {content_chunk}")
                    response_chunks.append(content_chunk)
        
        return ''.join(response_chunks)
    
    async def respond(self, message: 'Message') -> 'Message':
        """Receive a message and formulate a response.
        
        Will look up the message's Mondo history and use it for context
        before calling message() with the response.
        
        Args:
            message: The message to respond to
            
        Returns:
            Message: The response message
        """
        if not self.llm_client:
            raise ValueError(f"{self.__class__.__name__} requires an LLM client to formulate responses")
        
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
        
        # Get system prompt and generate response
        system_prompt = await self.get_system_prompt()
        llm_messages = [{"role": "system", "content": system_prompt}] + history
        response_content = await self.stream_llm_response(llm_messages)
        
        # Create and return the message
        return await self.message(mondo, response_content)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.name}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class Dojo:
    """Orchestrates the interaction between Ronin and Satori.
    
    The Dojo (道場) - "place of the way" - creates the space for learning
    and discovery. It manages the participants and initiates their dialogue.
    """
    
    def __init__(
        self,
        llm_client: Optional[AsyncOpenAI] = None,
        ronin_obj: Optional['Ronin'] = None,
        satori_obj: Optional['Satori'] = None
    ):
        """Initialize the Dojo with optional existing models.
        
        Args:
            llm_client: OpenAI client for LLM interactions
            ronin_obj: Optional existing Ronin model instance
            satori_obj: Optional existing Satori model instance
        """
        self.llm_client = llm_client or AsyncOpenAI()
        self.ronin_obj = ronin_obj
        self.satori_obj = satori_obj
        self.ronin: Optional['Ronin'] = None
        self.satori: Optional['Satori'] = None
        self.model: Optional[DojoModel] = None
        
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from OpenAI that might be wrapped in markdown.
        
        Args:
            response: Raw response from OpenAI
            
        Returns:
            str: Cleaned JSON string
        """
        # Remove markdown code blocks if present
        if response.startswith('```') and response.endswith('```'):
            # Extract content between first and last ```
            lines = response.split('\n')
            # Remove first and last lines (```)
            lines = lines[1:-1]
            # Remove language identifier if present (e.g., ```json)
            if lines[0].lower() in ['json', 'javascript']:
                lines = lines[1:]
            response = '\n'.join(lines)
        return response.strip()
        
    async def initialize(self) -> None:
        """Initialize the Dojo's cultural and philosophical context.
        
        This method establishes the foundational elements that will shape
        all interactions within the Dojo, including:
        - Theme and principles
        - System messages for personality shaping
        - Cultural context
        """
        # First, let the LLM contemplate the Dojo's essence
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": (
                    "You are a master of Japanese philosophical traditions, "
                    "particularly Zen Buddhism and its influence on martial arts, "
                    "arts, and ways of learning.\n\n"
                    "Create a philosophical framework for a Dojo (道場) - a place of learning "
                    "where a wandering seeker (Ronin) and enlightened guide (Satori) "
                    "will engage in dialogue.\n\n"
                    "Consider:\n"
                    "1. The atmosphere and theme that will foster deep learning\n"
                    "2. Core principles that should guide their interaction\n"
                    "3. How the Ronin should approach their seeking\n"
                    "4. How the Satori should guide and teach\n\n"
                    "Respond in JSON format with these keys:\n"
                    "- theme (string): The overarching atmosphere and focus\n"
                    "- principles (list): Core principles as short phrases\n"
                    "- ronin_system_message (string): Guidance for the Ronin's role\n"
                    "- satori_system_message (string): Guidance for the Satori's role"
                )
            }],
            stream=True
        )
        
        # Collect the response
        full_response = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    logger.info(f"Dojo contemplation chunk: {content_chunk}")
                    full_response.append(content_chunk)
        
        # Parse the contemplation
        raw_response = ''.join(full_response)
        cleaned_json = self._clean_json_response(raw_response)
        contemplation = json.loads(cleaned_json)
        
        # Create the Dojo model instance
        self.model = await DojoModel.objects.acreate(
            theme=contemplation['theme'],
            principles=contemplation['principles'],
            ronin_system_message=contemplation['ronin_system_message'],
            satori_system_message=contemplation['satori_system_message']
        )
        logger.info(f"Created Dojo {self.model.id} with theme: {self.model.theme}")
        
    async def prepare_ronin(
        self,
        style: Optional[str] = None
    ) -> 'Ronin':
        """Create or get a Ronin instance."""
        if not self.model:
            raise ValueError("Dojo must be initialized before preparing participants")
            
        # Create the controller instance
        from ronins.ronin import Ronin
        self.ronin = Ronin(
            name="",  # Will be set during meditation
            interests=[],  # Will be set during meditation
            style=style or "",  # Will be set during meditation if not provided
            model_obj=self.ronin_obj,
            llm_client=self.llm_client
        )
        
        # Let the Ronin discover their identity through meditation
        await self.ronin.prepare_self(self.model.ronin_system_message)
        return self.ronin
        
    async def prepare_satori(
        self,
        teaching_style: Optional[str] = None
    ) -> 'Satori':
        """Create or get a Satori instance."""
        if not self.model:
            raise ValueError("Dojo must be initialized before preparing participants")
            
        # Create the Satori instance
        from satoris.satori import Satori
        self.satori = Satori(
            name="",  # Will be set during meditation
            specialties=[],  # Will be set during meditation
            teaching_style=teaching_style or "",  # Will be set during meditation if not provided
            model_obj=self.satori_obj,
            llm_client=self.llm_client
        )
        
        # Let the Satori discover their identity through meditation
        await self.satori.prepare_self(self.model.satori_system_message)
        return self.satori
    
    async def mondo(self) -> 'Mondo':
        """Initiate a mondo dialogue between Ronin and Satori."""
        if not self.model:
            raise ValueError("Dojo must be initialized before beginning a Mondo")
        if not self.ronin or not self.satori:
            raise ValueError("Both Ronin and Satori must be prepared before beginning a Mondo")
        
        # Let the Ronin contemplate and name their quest
        logger.info(f"Ronin {self.ronin.name} contemplating quest...")
        quest_title = await self.ronin.contemplate_quest()
        logger.info(f"Quest title: {quest_title}")
        
        # Create the quest
        quest = await Quest.objects.acreate(
            ronin=self.ronin.model_obj,
            satori=self.satori.model_obj,
            title=quest_title
        )
        logger.info(f"Created quest: {quest.id} - {quest.title}")
        
        # Create the mondo
        dojo = await DojoModel.objects.acreate()
        mondo = await Mondo.objects.acreate(quest=quest, dojo=dojo)
        logger.info(f"Created mondo: {mondo.id}")
        
        # Generate the opening question (shomon)
        await self._shomon(mondo)
        
        return mondo

    async def continue_mondo(self, mondo: 'Mondo', max_exchanges: Optional[int] = None) -> None:
        """Continue the Mondo dialogue until the Ronin reaches understanding or max exchanges.
        
        Args:
            mondo: The Mondo dialogue to continue
            max_exchanges: Optional maximum number of exchanges (for testing)
        """
        if not self.ronin or not self.satori:
            raise ValueError("Both Ronin and Satori must be prepared before continuing a Mondo")
        
        exchanges = 0
        try:
            while True:
                # Get latest message
                latest_message = await sync_to_async(lambda: mondo.messages.latest())()
                content = await sync_to_async(lambda: latest_message.content)()
                
                # Let the Ronin contemplate their understanding
                ronin_decision = await self.ronin.contemplate_understanding(content)
                
                if ronin_decision["should_end"]:
                    logger.info(f"Ronin concludes: {ronin_decision['reason']}")
                    break
                
                # Determine next speaker based on last message
                if isinstance(latest_message, RoninMessage):
                    logger.info("Satori's turn to respond...")
                    await self.satori.respond(latest_message)
                else:
                    logger.info("Ronin's turn to respond...")
                    await self.ronin.respond(latest_message)
                
                exchanges += 1
                if max_exchanges is not None and exchanges >= max_exchanges:
                    logger.info(f"Mondo {mondo.id} reached maximum exchanges ({max_exchanges})")
                    break
                
                await asyncio.sleep(0)
                
        except asyncio.CancelledError:
            logger.info(f"Mondo {mondo.id} conversation loop cancelled")
        except Exception as e:
            logger.error(f"Error in Mondo {mondo.id} conversation loop: {e}")
            raise
    
    async def _shomon(self, mondo: 'Mondo') -> None:
        """Generate and record the Ronin's first question (初問).
        
        In Zen tradition, shomon (初問) represents the initial question
        a student asks when approaching a master, setting the direction
        for their dialogue and learning journey.
        
        Args:
            mondo: The mondo instance for this dialogue
        """
        # Use the Ronin's LLM interaction methods to generate the opening question
        system_prompt = (
            f"You are a Ronin named {self.ronin.name} with interests in "
            f"{', '.join(self.ronin.interests)} and a {self.ronin.style} "
            f"approach. You have named your quest: '{mondo.quest.title}'\n\n"
            "Generate a thoughtful opening question that begins your journey of understanding. "
            "Consider the depth of what you seek to learn and how your interests shape your inquiry."
        )
        
        # Get the generated question using the Ronin's stream method
        shomon_content = await self.ronin.stream_llm_response([{
            "role": "system",
            "content": system_prompt
        }])
        
        logger.info(f"Ronin {self.ronin.name} opens the Mondo with completed shomon:\n{shomon_content}")
        
        # Create the shomon message using the Ronin's message method
        await self.ronin.message(mondo, shomon_content)
        logger.info(f"Shomon message recorded for Mondo {mondo.id}")

    async def cleanup(self) -> None:
        """Clean up resources used by the Dojo.
        
        This should be called when you're done using the Dojo instance
        to ensure proper cleanup of resources like API clients.
        """
        if hasattr(self.llm_client, 'close'):
            await self.llm_client.close()
            
    async def __aenter__(self):
        """Support async context manager protocol."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        await self.cleanup()

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
    
    async def respond(self, message: 'Message') -> 'Message':
        """Receive a message and formulate a response.
        
        Will look up the message's Mondo history and use it for context
        before calling message() with the response.
        
        Args:
            message: The message to respond to
            
        Returns:
            Message: The response message
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Each being must define their own way of responding")

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
            
        # First, let the Ronin discover their identity through meditation
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": self.model.ronin_system_message
            }, {
                "role": "user",
                "content": (
                    "Through deep meditation, discover your identity as a Ronin:\n"
                    "1. Your name (a meaningful Japanese name)\n"
                    "2. Your three main interests in exploring the world\n"
                    "3. Your personal style and approach (if not already specified)\n\n"
                    "Consider historical wandering monks, scholars, and artists for inspiration.\n"
                    "Respond in JSON format with keys: name (string), interests (list), and style (string)"
                )
            }],
            stream=True
        )
        
        # Collect streamed response
        full_response = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    logger.info(f"Ronin meditation chunk: {content_chunk}")
                    full_response.append(content_chunk)
        
        # Clean and parse the response
        raw_response = ''.join(full_response)
        logger.info(f"Quest contemplation response:\n{raw_response}")
        
        cleaned_json = self._clean_json_response(raw_response)
        contemplation = json.loads(cleaned_json)
        
        # Create or get the DB record
        from ronins.models import Ronin as RoninModel
        if self.ronin_obj is None:
            self.ronin_obj = await RoninModel.objects.acreate(
                name=contemplation['name'],
                interests=contemplation['interests'],
                style=style or contemplation['style']
            )
        
        # Create the controller instance
        from ronins.ronin import Ronin
        self.ronin = Ronin(
            name=contemplation['name'],
            interests=contemplation['interests'],
            style=style or contemplation['style'],
            model_obj=self.ronin_obj,
            llm_client=self.llm_client
        )
        return self.ronin
        
    async def prepare_satori(
        self,
        teaching_style: Optional[str] = None
    ) -> 'Satori':
        """Create or get a Satori instance."""
        if not self.model:
            raise ValueError("Dojo must be initialized before preparing participants")
            
        # First, let the Satori discover their identity through meditation
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": self.model.satori_system_message
            }, {
                "role": "user",
                "content": (
                    "Through profound meditation, reveal your identity as a Satori:\n"
                    "1. Your name (a meaningful Japanese name)\n"
                    "2. Your three areas of specialty and deep understanding\n"
                    "3. Your natural approach to guiding others (if not already specified)\n\n"
                    "Consider historical Zen masters, teachers, and philosophers for inspiration.\n"
                    "Respond in JSON format with keys: name (string), specialties (list), and teaching_style (string)"
                )
            }],
            stream=True
        )
        
        # Collect streamed response
        full_response = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    logger.info(f"Satori meditation chunk: {content_chunk}")
                    full_response.append(content_chunk)
        
        # Clean and parse the response
        raw_response = ''.join(full_response)
        logger.info(f"Raw LLM response:\n{raw_response}")
        
        cleaned_json = self._clean_json_response(raw_response)
        logger.info(f"Cleaned JSON:\n{cleaned_json}")
        
        try:
            meditation = json.loads(cleaned_json)
            logger.info(f"Parsed meditation data:\n{json.dumps(meditation, indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise
            
        # Handle British English spelling if present
        if 'specialities' in meditation and 'specialties' not in meditation:
            logger.info("Converting British spelling 'specialities' to American 'specialties'")
            meditation['specialties'] = meditation.pop('specialities')
            
        # Log expected vs received keys
        expected_keys = {'name', 'specialties', 'teaching_style'}
        received_keys = set(meditation.keys())
        logger.info(f"Expected keys: {expected_keys}")
        logger.info(f"Received keys: {received_keys}")
        
        if expected_keys != received_keys:
            logger.warning(f"Missing keys: {expected_keys - received_keys}")
            logger.warning(f"Extra keys: {received_keys - expected_keys}")
        
        # Create or get the Django model instance
        from satoris.models import Satori as SatoriModel
        if self.satori_obj is None:
            try:
                self.satori_obj = await SatoriModel.objects.acreate(
                    name=meditation['name'],
                    specialties=meditation['specialties'],
                    teaching_style=teaching_style or meditation['teaching_style']
                )
            except KeyError as e:
                logger.error(f"Failed to access required key: {e}")
                raise
        
        # Create the Satori instance
        from satoris.satori import Satori
        self.satori = Satori(
            name=meditation['name'],
            specialties=meditation['specialties'],
            teaching_style=teaching_style or meditation['teaching_style'],
            model_obj=self.satori_obj,
            llm_client=self.llm_client
        )
        return self.satori
    
    async def mondo(self) -> 'Mondo':
        """Initiate a mondo dialogue between Ronin and Satori."""
        if not self.model:
            raise ValueError("Dojo must be initialized before beginning a Mondo")
        if not self.ronin or not self.satori:
            raise ValueError("Both Ronin and Satori must be prepared before beginning a Mondo")
        
        # Let the Ronin contemplate and name their quest
        logger.info(f"Ronin {self.ronin.name} contemplating quest...")
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": self.model.ronin_system_message
            }, {
                "role": "user",
                "content": (
                    f"You are a Ronin named {self.ronin.name} with interests in "
                    f"{', '.join(self.ronin.interests)} and a {self.ronin.style} "
                    "approach.\n\n"
                    "Through meditation, envision the quest you wish to undertake. "
                    "What profound question or exploration calls to you?\n\n"
                    "Name this quest in a way that reflects its depth and your seeking nature. "
                    "Consider the style of titles like 'In Search of Lost Wisdom' or 'The Path Through Ancient Gardens'.\n\n"
                    "Respond in JSON format with key: quest_title (string)"
                )
            }],
            stream=True
        )
        
        
        # Collect streamed response
        full_response = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    logger.info(f"Quest contemplation chunk: {content_chunk}")
                    full_response.append(content_chunk)
        
        # Clean and parse the response
        raw_response = ''.join(full_response)
        logger.info(f"Quest contemplation response:\n{raw_response}")
        
        cleaned_json = self._clean_json_response(raw_response)
        contemplation = json.loads(cleaned_json)
        quest_title = contemplation['quest_title']
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
        """Continue an existing mondo dialogue."""
        try:
            exchanges = 0
            
            while True:
                # Get the latest message and its attributes safely
                latest_message = await sync_to_async(lambda: mondo.messages.last())()
                if not latest_message:
                    logger.error("No messages found in mondo")
                    break
                
                content = await sync_to_async(lambda: latest_message.content)()
                author = await sync_to_async(lambda: latest_message.author)()
                author_name = await sync_to_async(lambda: author.name)()
                
                exchange_info = f"Exchange {exchanges + 1}"
                if max_exchanges is not None:
                    exchange_info += f"/{max_exchanges}"
                logger.info(exchange_info)
                logger.info(f"Latest message from {author_name}:\n{content}")
                
                # Ask LLM if the conversation has reached a natural conclusion
                logger.info("Checking for natural conclusion...")
                stream = await self.llm_client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[{
                        "role": "system",
                        "content": (
                            "You are a dialogue observer analyzing if a conversation has reached a natural conclusion.\n\n"
                            "Consider these factors:\n"
                            "1. A moment of understanding has been reached\n"
                            "2. The initial question has been thoroughly explored\n"
                            "3. A natural ending point has emerged\n\n"
                            "IMPORTANT: You must respond with ONLY a JSON object in this exact format:\n"
                            "{\n"
                            '  "should_end": false,\n'
                            '  "reason": "Explanation of why the conversation should or should not end"\n'
                            "}\n\n"
                            "Do not include any other text or explanation outside the JSON object."
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
                            logger.info(f"Conclusion check chunk: {content_chunk}")
                            full_response.append(content_chunk)
                
                try:
                    raw_response = ''.join(full_response)
                    cleaned_json = self._clean_json_response(raw_response)
                    conclusion = json.loads(cleaned_json)
                    logger.info(f"Conclusion check: {json.dumps(conclusion, indent=2)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse conclusion check response: {raw_response}")
                    raise
                
                if conclusion["should_end"]:
                    logger.info(f"Mondo concluding: {conclusion['reason']}")
                    break
                
                # Determine next speaker based on last message
                if isinstance(latest_message, RoninMessage):
                    logger.info("Satori's turn to respond...")
                    # Stream Satori's response
                    stream = await self.llm_client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=[{
                            "role": "system",
                            "content": (
                                f"You are Satori {self.satori.name}, a guide with expertise in "
                                f"{', '.join(self.satori.specialties)}. Your teaching style is: "
                                f"{self.satori.teaching_style}\n\n"
                                "Respond to this seeker's question with wisdom and insight."
                            )
                        }, {
                            "role": "user",
                            "content": content
                        }],
                        stream=True
                    )
                    
                    full_response = []
                    async for chunk in stream:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            content_chunk = chunk.choices[0].delta.content
                            if content_chunk:
                                logger.info(f"Satori response chunk: {content_chunk}")
                                full_response.append(content_chunk)
                    
                    response_content = ''.join(full_response)
                    response = await self.satori.message(mondo, response_content)
                    logger.info(f"Satori {self.satori.name} completed response")
                else:
                    logger.info("Ronin's turn to respond...")
                    # Stream Ronin's response
                    stream = await self.llm_client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=[{
                            "role": "system",
                            "content": (
                                f"You are Ronin {self.ronin.name}, interested in "
                                f"{', '.join(self.ronin.interests)} with a {self.ronin.style} "
                                "travel style.\n\n"
                                "Respond to this guidance with reflection and insight."
                            )
                        }, {
                            "role": "user",
                            "content": content
                        }],
                        stream=True
                    )
                    
                    full_response = []
                    async for chunk in stream:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            content_chunk = chunk.choices[0].delta.content
                            if content_chunk:
                                logger.info(f"Ronin response chunk: {content_chunk}")
                                full_response.append(content_chunk)
                    
                    response_content = ''.join(full_response)
                    response = await self.ronin.message(mondo, response_content)
                    logger.info(f"Ronin {self.ronin.name} completed response")
                
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
        # Use the LLM to generate an appropriate opening question
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": (
                    f"You are a Ronin named {self.ronin.name} with interests in "
                    f"{', '.join(self.ronin.interests)} and a {self.ronin.style} "
                    f"approach. You have named your quest: '{mondo.quest.title}'\n\n"
                    "Generate a thoughtful opening question that begins your journey of understanding. "
                    "Consider the depth of what you seek to learn and how your interests shape your inquiry."
                )
            }],
            stream=True
        )
        
        # Get the generated question
        logger.info(f"Ronin {self.ronin.name} composing shomon...")
        shomon_chunks = []
        async for chunk in stream:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    logger.info(f"Shomon chunk: {content_chunk}")
                    shomon_chunks.append(content_chunk)
        
        shomon_content = ''.join(shomon_chunks)
        logger.info(f"Ronin {self.ronin.name} opens the Mondo with completed shomon:\n{shomon_content}")
        
        # Create the shomon message
        from mondos.models import RoninMessage
        await RoninMessage.objects.acreate(
            mondo=mondo,
            content=shomon_content,
            author=self.ronin_obj
        )
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

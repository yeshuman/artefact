from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from openai import AsyncOpenAI
from asgiref.sync import sync_to_async
import json
import logging
import asyncio
import numpy as np
from dojo.models import Dojo as DojoModel
from quests.models import Quest
from mondos.models import Mondo, RoninMessage, SatoriMessage
from entities.models import EntityArchetype, EntityReference

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
        content = await message.acontent
        mondo = await sync_to_async(lambda: message.mondo)()
        
        # Get conversation history
        messages = await mondo.messages.all()
        history = []
        for msg in messages:
            msg_content = await msg.acontent
            msg_author = await msg.aauthor
            msg_author_name = await sync_to_async(lambda: msg_author.name)()
            is_satori_message = await sync_to_async(lambda: isinstance(msg, SatoriMessage))()
            role = "assistant" if is_satori_message else "user"
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
        """Clean up a JSON response from the LLM.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            str: The cleaned JSON string
        """
        # Remove any markdown code block markers
        response = response.replace('```json', '').replace('```', '').strip()
        
        # Remove any leading/trailing whitespace and newlines
        response = response.strip()
        
        # If the response starts with a newline, remove it
        if response.startswith('\n'):
            response = response[1:]
            
        # If the response ends with a newline, remove it
        if response.endswith('\n'):
            response = response[:-1]
            
        return response
        
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using OpenAI's embedding model."""
        response = await self.llm_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)

    async def initialize(self) -> None:
        """Initialize the Dojo's cultural and philosophical context."""
        # First, let the LLM contemplate the Dojo's essence
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": (
                    "You are a master of cultural synthesis, deeply versed in traditions, "
                    "philosophies, and ways of learning from all cultures and time periods.\n\n"
                    "Create a unique philosophical framework for a Dojo (道場) (a place of learning) "
                    "where a seeker and guide will engage in dialogue. You can draw from ANY "
                    "cultural tradition, time period, or philosophical system - ancient or modern, "
                    "Eastern or Western, traditional or contemporary.\n\n"
                    "Examples of themes you might choose:\n"
                    "- A Gen-Z coding bootcamp with meme culture and modern tech philosophy\n"
                    "- An ancient Greek symposium with Socratic dialogue\n"
                    "- A Zen monastery in medieval Japan\n"
                    "- A Silicon Valley startup's mentorship program\n"
                    "- A Renaissance Italian artist's workshop\n"
                    "- A contemporary African storytelling circle\n\n"
                    "Consider:\n"
                    "1. What unique atmosphere and theme will foster deep learning?\n"
                    "2. What principles should guide their interaction?\n"
                    "3. How should the seeker approach their learning?\n"
                    "4. How should the guide approach their teaching?\n\n"
                    "Respond in JSON format with these keys:\n"
                    "- theme (string): The overarching atmosphere and cultural context\n"
                    "- principles (list): Core principles as short phrases\n"
                    "- ronin_system_message (string): Guidance for the seeker's role\n"
                    "- satori_system_message (string): Guidance for the guide's role"
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
        
        # Bootstrap archetypes and references
        await self._bootstrap_archetypes()
        
        logger.info("Dojo initialization complete with archetypes and references")

    async def _bootstrap_archetypes(self) -> None:
        """Bootstrap initial entity archetypes and references based on dojo theme."""
        # Get theme-specific archetypes from LLM
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": (
                    f"Given this dojo's theme:\n{self.model.theme}\n\n"
                    f"And principles:\n" + "\n".join(f"- {p}" for p in self.model.principles) + "\n\n"
                    "Define the foundational entity types that will be important in dialogues.\n"
                    "For each archetype, provide:\n"
                    "- name: A hierarchical identifier (e.g., 'Concept.Philosophical', 'Location.Historical')\n"
                    "- description: What this type of entity represents\n"
                    "- examples: 5-10 canonical examples of this type\n"
                    "- relationships: How it relates to other types\n\n"
                    "Respond in JSON format with a list of archetypes."
                )
            }],
            stream=True
        )
        
        # Parse response and create archetypes
        response = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    response.append(content)
        
        archetypes = json.loads(self._clean_json_response(''.join(response)))
        
        # Create archetypes and references
        for archetype in archetypes:
            # Get embedding for archetype
            embedding = await self._get_embedding(
                f"{archetype['name']}: {archetype['description']}"
            )
            
            # Create archetype
            entity_archetype = await EntityArchetype.objects.acreate(
                name=archetype['name'],
                description=archetype['description'],
                embedding=embedding
            )
            logger.info(f"Created archetype: {entity_archetype.name}")
            
            # Create reference entities
            for example in archetype['examples']:
                # Get embedding for example
                example_embedding = await self._get_embedding(example)
                
                # Create reference
                await EntityReference.objects.acreate(
                    text=example,
                    archetype=entity_archetype,
                    embedding=example_embedding,
                    description=f"A canonical example of {archetype['name']}",
                    mondo=None  # Global reference
                )
                logger.info(f"Created reference entity: {example} ({entity_archetype.name})")

    async def prepare_ronin(
        self,
        style: Optional[str] = None
    ) -> 'Ronin':
        """Create or get a Ronin instance."""
        if not self.model:
            raise ValueError("Dojo must be initialized before preparing participants")
            
        # Create the Ronin model instance
        from ronins.models import Ronin as RoninModel
        self.ronin_obj = await RoninModel.objects.acreate(
            name="",  # Will be set during meditation
            interests=[],  # Will be set during meditation
            style=style or "",  # Will be set during meditation if not provided
            system_prompt=self.model.ronin_system_message
        )
        
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
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": self.model.ronin_system_message
            }, {
                "role": "user",
                "content": (
                    "Meditate on your identity as a seeker in this dojo. "
                    "Who are you? What interests drive you? What is your style of learning?\n\n"
                    "Respond in JSON format with:\n"
                    "- name (string): Your chosen name\n"
                    "- interests (list): Your key interests and motivations\n"
                    "- style (string): Your personal approach to learning"
                )
            }],
            stream=True
        )
        
        # Collect response
        meditation_text = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    meditation_text.append(content)
                    logger.info(f"Ronin meditation chunk: {content}")
        
        # Parse meditation results
        meditation = json.loads(self._clean_json_response(''.join(meditation_text)))
        
        # Update Ronin with meditation results
        self.ronin.name = meditation['name']
        self.ronin.interests = meditation['interests']
        self.ronin.style = meditation['style']
        
        # Update model
        self.ronin_obj.name = meditation['name']
        self.ronin_obj.interests = meditation['interests']
        self.ronin_obj.style = meditation['style']
        await sync_to_async(self.ronin_obj.save)()
        
        logger.info(f"Prepared Ronin: {self.ronin.name}")
        return self.ronin
        
    async def prepare_satori(
        self,
        style: Optional[str] = None
    ) -> 'Satori':
        """Create or get a Satori instance."""
        if not self.model:
            raise ValueError("Dojo must be initialized before preparing participants")
            
        # Create the Satori model instance
        from satoris.models import Satori as SatoriModel
        self.satori_obj = await SatoriModel.objects.acreate(
            name="",  # Will be set during meditation
            specialties=[],  # Will be set during meditation
            teaching_style=style or "",  # Will be set during meditation if not provided
            system_prompt=self.model.satori_system_message
        )
        
        # Create the controller instance
        from satoris.satori import Satori
        self.satori = Satori(
            name="",  # Will be set during meditation
            specialties=[],  # Will be set during meditation
            teaching_style=style or "",  # Will be set during meditation if not provided
            model_obj=self.satori_obj,
            llm_client=self.llm_client
        )
        
        # Let the Satori discover their identity through meditation
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": self.model.satori_system_message
            }, {
                "role": "user",
                "content": (
                    "Meditate on your identity as a guide in this dojo. "
                    "Who are you? What are your specialties? What is your teaching style?\n\n"
                    "Respond in JSON format with:\n"
                    "- name (string): Your chosen name\n"
                    "- specialties (list): Your areas of expertise and wisdom\n"
                    "- teaching_style (string): Your approach to guiding seekers"
                )
            }],
            stream=True
        )
        
        # Collect response
        meditation_text = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    meditation_text.append(content)
                    logger.info(f"Satori meditation chunk: {content}")
        
        # Parse meditation results
        meditation = json.loads(self._clean_json_response(''.join(meditation_text)))
        
        # Update Satori with meditation results
        self.satori.name = meditation['name']
        self.satori.specialties = meditation['specialties']
        self.satori.teaching_style = meditation['teaching_style']
        
        # Update model
        self.satori_obj.name = meditation['name']
        self.satori_obj.specialties = meditation['specialties']
        self.satori_obj.teaching_style = meditation['teaching_style']
        await sync_to_async(self.satori_obj.save)()
        
        logger.info(f"Prepared Satori: {self.satori.name}")
        return self.satori
    
    async def mondo(self) -> 'Mondo':
        """Create a new mondo and start the conversation."""
        if not self.initialized:
            raise ValueError("Dojo must be initialized before creating mondo")
        
        if not self.ronin or not self.satori:
            raise ValueError("Both Ronin and Satori must be prepared before creating mondo")
        
        # Create the mondo
        mondo = await Mondo.objects.acreate(
            quest=self.ronin.quest,
            theme=self.theme,
            principles=self.principles
        )
        logger.info(f"Created mondo: {mondo.id}")
        
        # Have Ronin ask the initial question
        await self.ronin.respond(mondo)
        logger.info("Ronin asked initial question")
        
        return mondo

    async def continue_mondo(self, mondo: 'Mondo', max_exchanges: Optional[int] = None) -> None:
        """Continue the mondo dialogue autonomously until completion."""
        try:
            exchange_count = 0
            while True:
                if max_exchanges and exchange_count >= max_exchanges:
                    logger.info(f"Reached maximum exchanges ({max_exchanges})")
                    break

                # Get the last message
                last_message = await sync_to_async(lambda: mondo.messages.last())()
                
                if not last_message:
                    logger.error("No messages found in mondo")
                    break
                    
                # If last message was from Satori, let Ronin respond
                if isinstance(last_message, SatoriMessage):
                    response = await self.ronin.respond(mondo, last_message)
                    if response.is_conclusion:
                        logger.info("Ronin has indicated conversation is complete")
                        break
                # If last message was from Ronin, let Satori respond
                else:
                    response = await self.satori.respond(mondo, last_message)
                    
                exchange_count += 1
                await asyncio.sleep(1)  # Prevent overwhelming the system
                
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
        messages = [{
            "role": "system",
            "content": await self.ronin.get_system_prompt()
        }, {
            "role": "user",
            "content": (
                f"You have named your quest: '{await sync_to_async(lambda: mondo.quest.title)()}'\n\n"
                "Generate a thoughtful opening question that begins your journey of understanding. "
                "Consider the depth of what you seek to learn and how your interests shape your inquiry."
            )
        }]
        
        # Get the generated question using the Ronin's stream method
        shomon_content = await self.ronin.stream_llm_response(messages)
        
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

    async def process_exchange(self, mondo: 'Mondo') -> bool:
        """Process a single exchange in the conversation between Ronin and Satori.
        
        Returns True if the conversation should continue, False if it has reached a natural conclusion.
        """
        if not self.initialized:
            raise ValueError("Dojo must be initialized before processing exchanges")
        
        if not self.ronin or not self.satori:
            raise ValueError("Both Ronin and Satori must be prepared before processing exchanges")
        
        # Get the last message from the mondo
        messages = await mondo.messages.all()
        last_message = messages[-1] if messages else None
        
        logger.info(f"Processing exchange. Last message type: {type(last_message).__name__ if last_message else 'None'}")
        
        # If no messages or last message was from Satori, Ronin should respond
        if not last_message or isinstance(last_message, SatoriMessage):
            logger.info("Ronin's turn to respond")
            response = await self.ronin.respond(mondo)
            logger.info(f"Ronin responded: {response.content[:100]}...")
            return True
        
        # If last message was from Ronin, Satori should respond
        elif isinstance(last_message, RoninMessage):
            logger.info("Satori's turn to respond")
            response = await self.satori.respond(mondo)
            logger.info(f"Satori responded: {response.content[:100]}...")
            return True
        
        else:
            logger.error(f"Unexpected message type: {type(last_message)}")
            return False

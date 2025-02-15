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
    """A seeker of knowledge in the dojo."""
    
    async def respond(self, mondo: 'Mondo', message: Optional['SatoriMessage'] = None) -> 'RoninMessage':
        """Generate a response in the conversation.
        
        If message is None, this is the initial question to start the conversation.
        """
        # Build conversation context
        messages = []
        
        # Add system message to guide the response
        if not message:
            # Initial question - add system message about asking first question
            messages.append({
                "role": "system",
                "content": (
                    f"You are {self.name}, a seeker of knowledge in the dojo. "
                    f"Your interests are: {self.interests}. Your communication style is: {self.style}. "
                    f"The dojo's theme is: {mondo.theme}. Its principles are: {mondo.principles}. "
                    "You are about to begin a philosophical dialogue with your guide. "
                    "Ask a profound opening question that aligns with the dojo's theme and principles, "
                    "drawing from your interests and expressing it in your unique style."
                )
            })
        else:
            # Regular response - add system message about continuing conversation
            messages.append({
                "role": "system", 
                "content": (
                    f"You are {self.name}, continuing a philosophical dialogue in the dojo. "
                    f"Your interests are: {self.interests}. Your communication style is: {self.style}. "
                    "Consider the guide's previous response carefully and respond in a way that "
                    "deepens the exploration of the topic while staying true to your character."
                )
            })
            
            # Add previous messages for context
            async for msg in mondo.messages.all():
                role = "assistant" if isinstance(msg, SatoriMessage) else "user"
                messages.append({"role": role, "content": msg.content})
        
        # Get response from LLM
        response = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=0.7,
            stream=True
        )
        
        # Collect response chunks
        content = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                
        # Create and return message
        return await RoninMessage.objects.acreate(
            mondo=mondo,
            content=content,
            author=self.model_obj
        )
        
    async def contemplate_quest(self) -> str:
        """Contemplate and name the quest being undertaken."""
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": self.model_obj.system_prompt
            }, {
                "role": "user",
                "content": (
                    f"As {self.name}, contemplate your quest in this dojo.\n"
                    "What is the title that best captures your seeking?"
                )
            }],
            stream=True
        )
        
        # Collect response
        title_parts = []
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    title_parts.append(content)
        
        return ''.join(title_parts).strip()

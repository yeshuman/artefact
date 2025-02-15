from django.db import models
from polymorphic.models import PolymorphicModel
from asgiref.sync import sync_to_async


class Mondo(models.Model):
    """A dialogue between Ronin and Satori within a quest.
    
    In the Zen tradition, a Mondō (問答) is a spiritual dialogue between
    seeker and master. Here, it represents the exchange between Ronin
    (the seeker) and Satori (the guide) in their journey of discovery.
    
    The Mondo serves as a container for messages that may contain entities
    and artefacts discovered during the dialogue. These entities are
    detected and processed asynchronously as the conversation unfolds,
    enriching the dialogue with structured knowledge and insights.
    
    Attributes:
        dojo: The Dojo context where this dialogue takes place
        quest: The specific Quest this dialogue belongs to
        created_at: Timestamp of when this dialogue began
    """
    dojo = models.ForeignKey('dojo.Dojo', on_delete=models.CASCADE, related_name='mondos')
    quest = models.ForeignKey('quests.Quest', on_delete=models.CASCADE, related_name='mondos')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Mondo in {self.quest.title}"


class Message(PolymorphicModel):
    """A message within a Mondo dialogue."""
    mondo = models.ForeignKey(Mondo, related_name='messages', on_delete=models.CASCADE)
    content = models.TextField()
    processed_content = models.TextField(
        null=True,
        blank=True,
        help_text="Content with detected entities marked up"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        get_latest_by = 'created_at'

    def __str__(self):
        return f"{self.content[:50]}..."

    @property
    async def acontent(self):
        """Async access to content field."""
        return await sync_to_async(lambda: self.content)()

    @property
    async def aprocessed_content(self):
        """Async access to processed_content field."""
        return await sync_to_async(lambda: self.processed_content)()

    @property
    async def aauthor(self):
        """Async access to author field."""
        return await sync_to_async(lambda: self.author)()

    @property
    def has_entities(self):
        """Check if the message has detected entities."""
        return bool(self.processed_content)

    @property
    async def aentities(self):
        """Async access to related entities."""
        return await self.entities.all()


class RoninMessage(Message):
    """A question or reflection from the Ronin."""
    author = models.ForeignKey('ronins.Ronin', on_delete=models.CASCADE, related_name='messages')

    @property
    async def process_entities(self):
        """Process entities in the Ronin's message.
        
        Ronin messages are processed for location-related entities that may
        indicate places of interest or travel destinations.
        """
        from .views import process_entities  # Import here to avoid circular imports
        await process_entities(self)


class SatoriMessage(Message):
    """An illuminating response from the Satori."""
    author = models.ForeignKey('satoris.Satori', on_delete=models.CASCADE, related_name='messages')

    @property
    async def process_entities(self):
        """Process entities in the Satori's response.
        
        Satori messages are processed for multiple entity types including:
        - Locations (temples, sacred sites)
        - Concepts (philosophical ideas)
        - Practices (meditation techniques)
        - People (historical figures, teachers)
        """
        from .views import process_entities  # Import here to avoid circular imports
        await process_entities(self)

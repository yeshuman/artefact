from django.db import models
from django.conf import settings
from django.contrib.postgres.indexes import Index
from pgvector.django import VectorField


class EntityArchetype(models.Model):
    """
    A high-level pattern or category that defines the fundamental nature of entities.
    Archetypes provide base embeddings and guide entity classification.
    
    Examples:
    - Location.Geographic (cities, mountains)
    - Object.Physical (ingredients, tools)
    - Process.Sequential (recipes, algorithms)
    - Agent.Person (historical figures)
    """
    name = models.CharField(
        max_length=100,
        help_text="The name of this archetype (e.g., 'Location.Geographic')"
    )
    description = models.TextField(
        help_text="Description of what this archetype represents and examples"
    )
    parent = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='children',
        help_text="Parent archetype if this is a sub-type"
    )
    embedding = VectorField(
        dimensions=1536,  # OpenAI ada-002 embedding size
        help_text="The vector embedding representing this archetype's essence"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['parent']),
            Index(name='archetype_embedding_idx', fields=['embedding'], opclasses=['vector_l2_ops']),
        ]
    
    def __str__(self):
        return self.name
    
    @property
    def full_path(self) -> str:
        """Get the full path of this archetype (e.g., 'Location.Geographic')."""
        if self.parent:
            return f"{self.parent.full_path}.{self.name}"
        return self.name

class EntityReference(models.Model):
    """
    A reference entity represents a canonical form of an entity that can be detected
    in conversations. It serves as a reference point for entity matching and linking.
    """
    text = models.TextField()
    type = models.CharField(max_length=50)
    embedding = VectorField(dimensions=1536)  # For OpenAI embeddings
    mondo = models.ForeignKey('mondos.Mondo', on_delete=models.CASCADE, related_name='entity_references')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            Index(name='reference_embedding_idx', fields=['embedding'], opclasses=['vector_l2_ops'])
        ]
        unique_together = ['text', 'type', 'mondo']

    def __str__(self):
        return f"{self.type}: {self.text}"

class Entity(models.Model):
    """
    Represents an entity detected in a message.
    Entities are matched against reference entities for classification.
    """
    text = models.CharField(max_length=255, help_text="The actual text of the entity")
    archetype = models.ForeignKey(
        EntityArchetype,
        on_delete=models.PROTECT,
        related_name='detected_entities',
        help_text="The archetype this entity belongs to"
    )
    confidence = models.FloatField(
        help_text="Confidence score of the entity detection"
    )
    start_position = models.IntegerField(
        help_text="Starting position of the entity in the message text"
    )
    end_position = models.IntegerField(
        help_text="Ending position of the entity in the message text"
    )
    message = models.ForeignKey(
        'mondos.Message',
        on_delete=models.CASCADE,
        related_name='entities',
        help_text="The message containing this entity"
    )
    reference_entity = models.ForeignKey(
        EntityReference,
        on_delete=models.SET_NULL,
        null=True,
        related_name='detected_entities',
        help_text="The reference entity this was matched against"
    )
    embedding = VectorField(
        dimensions=1536,  # OpenAI ada-002 embedding size
        help_text="The vector embedding for this entity instance"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "entities"
        ordering = ['-confidence', 'created_at']
        indexes = [
            models.Index(fields=['text']),
            models.Index(fields=['archetype']),
            models.Index(fields=['message']),
            models.Index(fields=['confidence']),
            Index(name='entity_embedding_idx', fields=['embedding'], opclasses=['vector_l2_ops']),
        ]
    
    def __str__(self):
        return f"{self.text} ({self.archetype.name})"
    
    @property
    def length(self):
        """Return the length of the entity text."""
        return self.end_position - self.start_position
    
    @property
    def is_confident(self):
        """Check if the entity meets the confidence threshold."""
        threshold = getattr(settings, 'ENTITY_CONFIDENCE_THRESHOLD', 0.8)
        return self.confidence >= threshold

from django.db import models
from django.conf import settings
from pgvector.django import VectorField, IvfflatIndex, HnswIndex
from django.db import connection
from asgiref.sync import sync_to_async
from django.contrib.postgres.fields import ArrayField


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
            HnswIndex(
                name='archetype_embedding_idx',
                fields=['embedding'],
                opclasses=['vector_l2_ops'],
                m=16,
                ef_construction=64,
            )
        ]
    
    def __str__(self):
        return self.name
    
    @property
    def full_path(self) -> str:
        """Get the full path of this archetype (e.g., 'Location.Geographic')."""
        if self.parent:
            return f"{self.parent.full_path}.{self.name}"
        return self.name
    
    @classmethod
    async def is_duplicate(cls, name: str, embedding) -> bool:
        """
        Check if an archetype with the same name or very similar embedding exists.
        
        Args:
            name: The name to check for duplicates
            embedding: The embedding to check for similarity
            
        Returns:
            bool: True if a duplicate exists, False otherwise
        """
        # Check for exact name match
        name_exists = await cls.objects.filter(name=name).aexists()
        if name_exists:
            return True
            
        # Check for similar embeddings using raw SQL
        # Convert embedding to string for SQL
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        query = f"""
            SELECT EXISTS (
                SELECT 1 FROM entities_entityarchetype
                WHERE embedding <-> %s::vector < 0.05
            )
        """
        
        @sync_to_async
        def check_similar():
            with connection.cursor() as cursor:
                cursor.execute(query, [embedding_str])
                result = cursor.fetchone()
                return result[0]
        
        return await check_similar()

    @property
    async def adescription(self):
        """Async access to description field."""
        return self.description

    @property
    async def aembedding(self):
        """Async access to embedding field."""
        return self.embedding

class EntityReference(models.Model):
    """
    A reference entity represents a canonical form of an entity that can be detected
    in conversations. It serves as a reference point for entity matching and linking.
    """
    text = models.TextField()
    archetype = models.ForeignKey(
        EntityArchetype,
        on_delete=models.PROTECT,
        related_name='reference_entities',
        help_text="The archetype this reference entity belongs to"
    )
    description = models.TextField(
        null=True,
        blank=True,
        help_text="Description of this reference entity"
    )
    embedding = VectorField(dimensions=1536)  # For OpenAI embeddings
    mondo = models.ForeignKey('mondos.Mondo', on_delete=models.CASCADE, related_name='entity_references', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['text', 'archetype', 'mondo']
        indexes = [
            models.Index(fields=['text']),
            models.Index(fields=['archetype']),
            HnswIndex(
                name='reference_embedding_idx',
                fields=['embedding'],
                opclasses=['vector_l2_ops'],
                m=16,
                ef_construction=64,
            )
        ]

    def __str__(self):
        return f"{self.archetype.name}: {self.text}"

    @property
    async def aembedding(self):
        """Async access to embedding field."""
        return self.embedding

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
            HnswIndex(
                name='entity_embedding_idx',
                fields=['embedding'],
                opclasses=['vector_l2_ops'],
                m=16,
                ef_construction=64,
            )
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

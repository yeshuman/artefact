from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.conf import settings
from pgvector.django import VectorField, HnswIndex
from polymorphic.models import PolymorphicModel
from typing import List


class Artefact(models.Model):
    """
    An enriched object derived from detected entities.
    Artefacts are enhanced with data from external APIs and maintain
    their relationship to source entities.
    """
    
    # Core Fields
    name = models.CharField(
        max_length=255,
        help_text="The canonical name of this artefact"
    )
    slug = models.SlugField(
        unique=True,
        help_text="URL-friendly version of the name"
    )
    description = models.TextField(
        null=True,
        blank=True,
        help_text="Rich description combining various data sources"
    )
    
    # Relationships
    source_entity = models.ForeignKey(
        'entities.Entity',
        on_delete=models.SET_NULL,
        null=True,
        related_name='artefacts',
        help_text="The entity this artefact was derived from"
    )
    
    # Vector Embedding
    embedding = VectorField(
        dimensions=1536,  # OpenAI ada-002 embedding size
        null=True,
        help_text="Vector embedding for similarity search"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "artefact"
        verbose_name_plural = "artefacts"
        ordering = ['-created_at']
        indexes = [
            HnswIndex(
                name='artefact_embedding_idx',
                fields=['embedding'],
                opclasses=['vector_l2_ops'],
                m=16,
                ef_construction=64,
            )
        ]
    
    def __str__(self):
        return f"{self.name} ({self.source_entity.archetype.name if self.source_entity else 'Unknown'})"
    
    @property
    def mondo(self):
        """Get the mondo this artefact belongs to through its source entity."""
        return self.source_entity.message.mondo if self.source_entity else None
    
    @property
    def has_any_data(self):
        """Check if the artefact has any enriched data."""
        return self.data_sources.exists()
    
    @property
    def data_sources_status(self):
        """Get a summary of data source states."""
        from django.db.models import Count
        return self.enrichment_tasks.values('data_type', 'state').annotate(count=Count('id'))

    async def get_similar_artefacts(self, threshold: float = 0.2, limit: int = 10):
        """Find artefacts with similar embeddings."""
        if not self.embedding:
            return []
        
        return await Artefact.objects.filter(
            embedding__cosine_distance__lte=threshold
        ).exclude(
            id=self.id
        ).order_by(
            'embedding__cosine_distance'
        )[:limit].aall()
    
    async def get_nearby_artefacts(self, distance_km: float = 1.0, limit: int = 10):
        """Find artefacts within specified distance."""
        location = await ArtefactLocation.objects.filter(artefact=self).afirst()
        if not location:
            return []
        
        nearby_locations = await ArtefactLocation.objects.filter(
            latitude__range=(location.latitude - 0.01 * distance_km, 
                           location.latitude + 0.01 * distance_km),
            longitude__range=(location.longitude - 0.01 * distance_km,
                            location.longitude + 0.01 * distance_km)
        ).exclude(
            artefact=self
        ).select_related(
            'artefact'
        )[:limit].aall()
        
        return [loc.artefact for loc in nearby_locations]
    
    async def get_temporal_overlaps(self, limit: int = 10):
        """Find artefacts with overlapping time periods."""
        context = await ArtefactTemporalContext.objects.filter(artefact=self).afirst()
        if not context or not (context.start_date or context.end_date):
            return []
        
        query = ArtefactTemporalContext.objects.exclude(artefact=self)
        
        if context.start_date and context.end_date:
            query = query.filter(
                models.Q(start_date__lte=context.end_date) &
                models.Q(end_date__gte=context.start_date)
            )
        elif context.start_date:
            query = query.filter(end_date__gte=context.start_date)
        else:
            query = query.filter(start_date__lte=context.end_date)
        
        overlapping = await query.select_related(
            'artefact'
        )[:limit].aall()
        
        return [ctx.artefact for ctx in overlapping]
    
    async def get_wikidata_relations(
        self,
        property_id: str = None,
        min_confidence: float = 0.5,
        include_inverse: bool = True,
        limit: int = 10
    ) -> List['Artefact']:
        """
        Find artefacts related through Wikidata properties.
        
        Args:
            property_id: Optional Wikidata property ID to filter by (e.g., 'P131')
            min_confidence: Minimum confidence score for relationships
            include_inverse: Whether to include inverse relationships
            limit: Maximum number of results to return
        """
        query = ArtefactWikidataRelation.objects.filter(
            artefact=self,
            confidence__gte=min_confidence
        )
        
        if property_id:
            query = query.filter(property_id=property_id)
        
        if not include_inverse:
            query = query.filter(is_inverse=False)
        
        relations = await query.select_related(
            'related_artefact'
        ).order_by(
            '-confidence'
        )[:limit].aall()
        
        return [rel.related_artefact for rel in relations]
    
    async def get_all_relations(self, limit_per_type: int = 5):
        """Get all types of relations for this artefact."""
        similar = await self.get_similar_artefacts(limit=limit_per_type)
        nearby = await self.get_nearby_artefacts(limit=limit_per_type)
        temporal = await self.get_temporal_overlaps(limit=limit_per_type)
        wikidata = await self.get_wikidata_relations(limit=limit_per_type)
        
        return {
            'similar': similar,
            'nearby': nearby,
            'temporal': temporal,
            'wikidata': wikidata
        }


class ArtefactData(PolymorphicModel):
    """
    Base model for artefact data from external sources.
    Each source (Google Places, Wikipedia, etc.) will have its own
    structured data model inheriting from this.
    """
    artefact = models.ForeignKey(
        Artefact,
        on_delete=models.CASCADE,
        related_name='data_sources',
        help_text="The artefact this data belongs to"
    )
    fetched_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this data was fetched from the source"
    )
    raw_data = models.JSONField(
        help_text="The complete raw data from the source"
    )

    class Meta:
        ordering = ['-fetched_at']
        indexes = [
            models.Index(fields=['artefact', 'fetched_at'])
        ]

    def __str__(self):
        return f"{self.__class__.__name__} for {self.artefact.name}"


class ArtefactGooglePlaceData(ArtefactData):
    """Google Places API data for an artefact."""
    place_id = models.CharField(
        max_length=255,
        unique=True,
        help_text="Google Places unique identifier"
    )
    rating = models.FloatField(
        null=True,
        blank=True,
        help_text="Google rating (0-5)"
    )
    rating_count = models.IntegerField(
        null=True,
        blank=True,
        help_text="Number of ratings"
    )
    price_level = models.IntegerField(
        null=True,
        blank=True,
        help_text="Price level (1-4)"
    )
    
    class Meta:
        verbose_name = "Google Places data"
        verbose_name_plural = "Google Places data"
        indexes = [
            models.Index(fields=['place_id']),
            models.Index(fields=['rating'])
        ]


class ArtefactWikipediaData(ArtefactData):
    """Wikipedia article data for an artefact."""
    page_id = models.IntegerField(
        unique=True,
        help_text="Wikipedia page ID"
    )
    url = models.URLField(
        help_text="URL to the Wikipedia article"
    )
    summary = models.TextField(
        help_text="Article summary/extract"
    )
    last_updated = models.DateTimeField(
        help_text="When the Wikipedia article was last updated"
    )
    
    class Meta:
        verbose_name = "Wikipedia data"
        verbose_name_plural = "Wikipedia data"
        indexes = [
            models.Index(fields=['page_id'])
        ]


class ArtefactWikidataData(ArtefactData):
    """Wikidata structured data for an artefact."""
    entity_id = models.CharField(
        max_length=20,
        unique=True,
        help_text="Wikidata Q identifier"
    )
    wikidata_types = ArrayField(
        models.CharField(max_length=20),
        help_text="List of Wikidata Q IDs this entity is an instance of"
    )
    properties = models.JSONField(
        help_text="Structured Wikidata properties"
    )
    
    class Meta:
        verbose_name = "Wikidata"
        verbose_name_plural = "Wikidata"
        indexes = [
            models.Index(fields=['entity_id'])
        ]


class ArtefactTripAdvisorData(ArtefactData):
    """TripAdvisor data for an artefact."""
    location_id = models.CharField(
        max_length=50,
        unique=True,
        help_text="TripAdvisor location identifier"
    )
    rating = models.FloatField(
        null=True,
        blank=True,
        help_text="TripAdvisor rating (1-5)"
    )
    review_count = models.IntegerField(
        null=True,
        blank=True,
        help_text="Number of reviews"
    )
    ranking = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Ranking in category (e.g., '#5 of 100 restaurants in Tokyo')"
    )
    ranking_data = models.JSONField(
        null=True,
        blank=True,
        help_text="Structured ranking information"
    )
    awards = ArrayField(
        models.JSONField(),
        default=list,
        blank=True,
        help_text="TripAdvisor awards (e.g., Travelers' Choice)"
    )
    
    class Meta:
        verbose_name = "TripAdvisor data"
        verbose_name_plural = "TripAdvisor data"
        indexes = [
            models.Index(fields=['location_id']),
            models.Index(fields=['rating'])
        ]


class ArtefactAttribute(PolymorphicModel):
    """
    Base model for artefact attributes. Child models will define specific
    attribute types with their own fields and validation.
    
    This is a polymorphic model, allowing us to query all attributes together
    while maintaining type-specific behavior.
    """
    artefact = models.ForeignKey(
        Artefact,
        on_delete=models.CASCADE,
        related_name='attributes',
        help_text="The artefact this attribute belongs to"
    )
    source = models.CharField(
        max_length=50,
        help_text="The source of this attribute data (e.g., 'google_places', 'wikipedia')"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['artefact', 'source'])
        ]


class ArtefactLocation(ArtefactAttribute):
    """Geographic location information for an artefact."""
    latitude = models.FloatField(
        help_text="Latitude coordinate"
    )
    longitude = models.FloatField(
        help_text="Longitude coordinate"
    )
    address = models.JSONField(
        null=True,
        blank=True,
        help_text="Structured address components"
    )

    class Meta:
        indexes = [
            models.Index(fields=['latitude', 'longitude'])
        ]

    def __str__(self):
        return f"Location for {self.artefact.name}: ({self.latitude}, {self.longitude})"


class ArtefactTemporalContext(ArtefactAttribute):
    """Temporal information about an artefact."""
    start_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the artefact began existing or event started"
    )
    end_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the artefact ceased existing or event ended"
    )
    period_name = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        help_text="Named period (e.g., 'Edo Period', 'Renaissance')"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['start_date', 'end_date'])
        ]

    def __str__(self):
        return f"Temporal context for {self.artefact.name}"


class ArtefactContact(ArtefactAttribute):
    """Contact and operational information."""
    website = models.URLField(
        null=True,
        blank=True
    )
    phone = models.CharField(
        max_length=50,
        null=True,
        blank=True
    )
    email = models.EmailField(
        null=True,
        blank=True
    )
    hours = models.JSONField(
        null=True,
        blank=True,
        help_text="Operating hours structure"
    )

    def __str__(self):
        return f"Contact info for {self.artefact.name}"


class ArtefactMedia(ArtefactAttribute):
    """Media assets related to the artefact."""
    url = models.URLField(
        help_text="URL to the media asset"
    )
    type = models.CharField(
        max_length=20,
        choices=[
            ('IMAGE', 'Image'),
            ('VIDEO', 'Video'),
            ('AUDIO', 'Audio'),
            ('DOCUMENT', 'Document')
        ]
    )
    title = models.CharField(
        max_length=255,
        null=True,
        blank=True
    )
    description = models.TextField(
        null=True,
        blank=True
    )
    metadata = models.JSONField(
        null=True,
        blank=True,
        help_text="Additional metadata about the media"
    )

    class Meta:
        indexes = [
            models.Index(fields=['type'])
        ]
        verbose_name_plural = "artefact media"

    def __str__(self):
        return f"{self.type} for {self.artefact.name}: {self.title or self.url}"


class ArtefactEnrichmentTask(models.Model):
    """
    Background task for enriching artefacts with API data.
    Each task represents a single attempt to fetch and create/update
    a specific type of ArtefactData.
    """
    
    artefact = models.ForeignKey(
        Artefact,
        on_delete=models.CASCADE,
        related_name='enrichment_tasks',
        help_text="The artefact to be enriched"
    )
    data_type = models.CharField(
        max_length=50,
        choices=[
            ('ArtefactGooglePlaceData', 'Google Places'),
            ('ArtefactWikipediaData', 'Wikipedia'),
            ('ArtefactWikidataData', 'Wikidata'),
            ('ArtefactTripAdvisorData', 'TripAdvisor')
        ],
        help_text="The type of ArtefactData to create/update"
    )
    state = models.CharField(
        max_length=20,
        choices=[
            ('PENDING', 'Task Pending'),
            ('IN_PROGRESS', 'Task Running'),
            ('COMPLETED', 'Task Completed'),
            ('FAILED', 'Task Failed')
        ],
        default='PENDING',
        help_text="Current state of the enrichment task"
    )
    error_message = models.TextField(
        null=True,
        blank=True,
        help_text="Error message if the task failed"
    )
    retry_count = models.IntegerField(
        default=0,
        help_text="Number of times this task has been retried"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the task started running"
    )
    completed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the task completed or failed"
    )
    result_data = models.ForeignKey(
        ArtefactData,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='creation_task',
        help_text="The ArtefactData instance created/updated by this task"
    )
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['state', 'created_at']),
            models.Index(fields=['artefact', 'data_type']),
            models.Index(fields=['state', 'retry_count'])
        ]
        unique_together = [
            ('artefact', 'data_type', 'state'),  # Only one active task per data type
        ]
    
    def __str__(self):
        return f"{self.data_type} enrichment for {self.artefact.name}"
    
    @property
    def can_retry(self):
        """Check if the task can be retried based on retry limits."""
        max_retries = getattr(settings, 'ENRICHMENT_MAX_RETRIES', 3)
        return self.state == 'FAILED' and self.retry_count < max_retries
    
    @property
    def data_model(self):
        """Get the model class for this task's data type."""
        from django.apps import apps
        return apps.get_model('artefacts', self.data_type)


class ArtefactWikidataRelation(ArtefactAttribute):
    """
    Structured relationships between artefacts based on Wikidata properties.
    Instead of hardcoding relationship types, we store the actual Wikidata
    property that establishes the relationship, allowing for dynamic
    relationship discovery based on the entity types we encounter.
    """
    property_id = models.CharField(
        max_length=20,
        help_text="The Wikidata property ID that establishes this relationship (e.g., 'P131')"
    )
    property_label = models.CharField(
        max_length=100,
        help_text="Human-readable label for the property (e.g., 'located in administrative territory')"
    )
    related_artefact = models.ForeignKey(
        'artefacts.Artefact',
        on_delete=models.CASCADE,
        related_name='wikidata_relations_to',
        help_text="The artefact this one is related to"
    )
    is_inverse = models.BooleanField(
        default=False,
        help_text="Whether this is an inverse relationship"
    )
    confidence = models.FloatField(
        help_text="Confidence score for this relationship (0-1)"
    )
    metadata = models.JSONField(
        null=True,
        blank=True,
        help_text="Additional context about the relationship"
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['property_id']),
            models.Index(fields=['confidence'])
        ]
    
    def __str__(self):
        direction = "<-" if self.is_inverse else "->"
        return f"{self.artefact.name} {direction} {self.property_label} {direction} {self.related_artefact.name}"

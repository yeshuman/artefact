import asyncio
from django.core.management.base import BaseCommand
from openai import AsyncOpenAI
from django.conf import settings
from entities.models import ReferenceEntity, Entity

class Command(BaseCommand):
    help = 'Create reference entities with their embeddings'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def get_embedding(self, text: str):
        """Get embedding vector for text using OpenAI's API."""
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    async def create_reference_entity(self, text: str, entity_type: str):
        """Create a reference entity with its embedding."""
        # Check if it already exists
        if await ReferenceEntity.objects.filter(text=text, type=entity_type).aexists():
            self.stdout.write(f"Reference entity already exists: {text} ({entity_type})")
            return
        
        # Get embedding
        embedding = await self.get_embedding(text)
        
        # Create reference entity
        ref_entity = await ReferenceEntity.objects.acreate(
            text=text,
            type=entity_type
        )
        ref_entity.set_embedding_array(embedding)
        await ref_entity.asave()
        
        self.stdout.write(f"Created reference entity: {text} ({entity_type})")
    
    async def create_initial_references(self):
        """Create initial set of reference entities."""
        references = {
            Entity.PLACE: [
                "Kyoto", "Tokyo", "Osaka", "Nara",
                "Kamakura", "Hakone", "Nikko"
            ],
            Entity.LANDMARK: [
                "Mount Fuji", "Mount Hiei", "Mount Koya",
                "Lake Biwa", "Arashiyama"
            ],
            Entity.SITE: [
                "Kinkaku-ji", "Ginkaku-ji", "Kiyomizu-dera",
                "Ryoan-ji Temple", "Todai-ji", "Fushimi Inari Shrine",
                "Meiji Shrine", "Sensoji Temple"
            ]
        }
        
        for entity_type, entities in references.items():
            for entity in entities:
                await self.create_reference_entity(entity, entity_type)
    
    def handle(self, *args, **options):
        """Run the command."""
        self.stdout.write("Creating reference entities...")
        asyncio.run(self.create_initial_references())
        self.stdout.write(self.style.SUCCESS("Successfully created reference entities")) 
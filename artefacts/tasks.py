"""
Background tasks for artefact enrichment.
"""
from django.utils import timezone
from django.db import transaction
from .models import ArtefactEnrichmentTask, ArtefactWikidataData
from .services.wikidata import process_wikidata_relationships

async def process_wikidata_enrichment(task: ArtefactEnrichmentTask) -> None:
    """
    Process a Wikidata enrichment task.
    1. Fetch raw Wikidata data
    2. Store it in ArtefactWikidataData
    3. Process relationships from the data
    """
    try:
        # Update task state
        task.state = 'IN_PROGRESS'
        task.started_at = timezone.now()
        await task.asave()
        
        # Get Wikidata client
        client = get_wikidata_client()
        
        # Fetch and store data
        async with transaction.atomic():
            # Fetch raw data
            data = await client.fetch_entity_data(task.artefact.name)
            
            # Store raw data
            wikidata = await ArtefactWikidataData.objects.acreate(
                artefact=task.artefact,
                entity_id=data['id'],
                wikidata_types=data['instance_of'],
                properties=data['properties'],
                raw_data=data
            )
            
            # Process relationships
            await process_wikidata_relationships(task.artefact)
            
            # Update task
            task.state = 'COMPLETED'
            task.completed_at = timezone.now()
            task.result_data = wikidata
            await task.asave()
            
    except Exception as e:
        # Handle failure
        task.state = 'FAILED'
        task.error_message = str(e)
        task.completed_at = timezone.now()
        await task.asave() 
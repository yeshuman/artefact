"""
Service for processing Wikidata data into structured artefact relationships.
"""
from typing import Dict, List, Set, Tuple
from django.db import transaction
from artefacts.models import (
    Artefact,
    ArtefactWikidataData,
    ArtefactWikidataRelation
)

# Properties that can have inverse relationships
INVERSE_PROPERTIES = {
    'P361': 'P527',  # part of <-> has part
    'P527': 'P361',  # has part <-> part of
    'P7': 'P6',      # brother <-> sister
    'P6': 'P7',      # sister <-> brother
    # Add more as we discover them
}

# Properties that are inherently bidirectional
BIDIRECTIONAL_PROPERTIES = {
    'P460',  # said to be the same as
    'P1889', # different from
    'P1676', # related to
    # Add more as we discover them
}

async def process_wikidata_relationships(artefact: Artefact) -> None:
    """
    Process an artefact's Wikidata data to establish relationships with other artefacts.
    Instead of using predefined mappings, we discover relationships from the data.
    """
    wikidata = await ArtefactWikidataData.objects.filter(artefact=artefact).afirst()
    if not wikidata or not wikidata.properties:
        return
    
    # Process each property in the data
    async with transaction.atomic():
        for prop_id, values in wikidata.properties.items():
            # Skip non-Q-ID properties
            if not isinstance(values, (list, str)) or not any(
                isinstance(v, str) and v.startswith('Q') 
                for v in (values if isinstance(values, list) else [values])
            ):
                continue
            
            # Get property label from Wikidata
            prop_label = await get_property_label(prop_id)
            if not prop_label:
                continue
            
            # Find related Q IDs
            related_qids = set(
                v for v in (values if isinstance(values, list) else [values])
                if isinstance(v, str) and v.startswith('Q')
            )
            
            # Find artefacts with these Q IDs
            related_artefacts = await _find_artefacts_by_qids(related_qids)
            
            # Calculate confidence based on property type and data quality
            confidence = await calculate_relationship_confidence(
                prop_id, wikidata.wikidata_types
            )
            
            # Create relationships
            for related in related_artefacts:
                # Create forward relationship
                await ArtefactWikidataRelation.objects.acreate(
                    artefact=artefact,
                    related_artefact=related,
                    property_id=prop_id,
                    property_label=prop_label,
                    is_inverse=False,
                    confidence=confidence,
                    source='wikidata',
                    metadata={
                        'wikidata_types': wikidata.wikidata_types,
                        'property_values': values
                    }
                )
                
                # Handle bidirectional properties
                if prop_id in BIDIRECTIONAL_PROPERTIES:
                    await ArtefactWikidataRelation.objects.acreate(
                        artefact=related,
                        related_artefact=artefact,
                        property_id=prop_id,
                        property_label=prop_label,
                        is_inverse=False,
                        confidence=confidence,
                        source='wikidata',
                        metadata={
                            'wikidata_types': wikidata.wikidata_types,
                            'property_values': values
                        }
                    )
                
                # Handle inverse properties
                elif prop_id in INVERSE_PROPERTIES:
                    inverse_prop_id = INVERSE_PROPERTIES[prop_id]
                    inverse_label = await get_property_label(inverse_prop_id)
                    if inverse_label:
                        await ArtefactWikidataRelation.objects.acreate(
                            artefact=related,
                            related_artefact=artefact,
                            property_id=inverse_prop_id,
                            property_label=inverse_label,
                            is_inverse=True,
                            confidence=confidence,
                            source='wikidata',
                            metadata={
                                'wikidata_types': wikidata.wikidata_types,
                                'property_values': values
                            }
                        )

async def calculate_relationship_confidence(prop_id: str, wikidata_types: List[str]) -> float:
    """
    Calculate confidence score for a relationship based on:
    1. Property usage frequency for these instance types
    2. Property quality metrics from Wikidata
    3. Historical accuracy of this property in our system
    """
    # For now, return a default confidence
    # TODO: Implement proper confidence calculation
    return 0.8

async def get_property_label(prop_id: str) -> str:
    """Get human-readable label for a Wikidata property."""
    # TODO: Implement property label fetching from Wikidata
    # For now, return the property ID
    return prop_id

async def _find_artefacts_by_qids(qids: Set[str]) -> List[Artefact]:
    """Find artefacts that have Wikidata data with the given Q IDs."""
    wikidata = await ArtefactWikidataData.objects.filter(
        entity_id__in=qids
    ).select_related(
        'artefact'
    ).aall()
    
    return [data.artefact for data in wikidata] 
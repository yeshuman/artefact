import numpy as np
from typing import List, Dict, Tuple, Optional
from django.conf import settings
from openai import AsyncOpenAI
import logging
from asgiref.sync import sync_to_async

from .models import Entity, ReferenceEntity

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for text using OpenAI's API."""
    try:
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        logger.error(f"Error getting embedding for '{text}': {e}")
        return np.zeros(1536)  # Default embedding dimension for ada-002

async def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

@sync_to_async
def get_mondo_reference_entities(mondo) -> List[ReferenceEntity]:
    """Get reference entities for a specific mondo."""
    return list(ReferenceEntity.objects.filter(mondo=mondo))

async def find_best_match(text_embedding: np.ndarray, mondo) -> Tuple[Optional[ReferenceEntity], float]:
    """
    Find the best matching reference entity for a given embedding within a mondo.
    Returns the reference entity and the similarity score.
    """
    reference_entities = await get_mondo_reference_entities(mondo)
    best_match = None
    best_similarity = 0.0
    
    for ref_entity in reference_entities:
        ref_embedding = ref_entity.get_embedding_array()
        similarity = await calculate_similarity(text_embedding, ref_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = ref_entity
    
    return best_match, best_similarity

async def create_reference_entity(text: str, entity_type: str, mondo) -> ReferenceEntity:
    """Create a new reference entity for the mondo."""
    embedding = await get_embedding(text)
    
    ref_entity = await ReferenceEntity.objects.acreate(
        text=text,
        type=entity_type,
        mondo=mondo
    )
    ref_entity.set_embedding_array(embedding)
    await sync_to_async(ref_entity.save)()
    
    logger.info(f"Created reference entity: {text} ({entity_type}) for mondo {mondo.id}")
    return ref_entity

async def detect_entities(message) -> List[Entity]:
    """
    Main entity detection function that processes a message and returns detected entities.
    Compares potential entities against known reference entities within the same mondo.
    """
    logger.info(f"Processing message: {message.content[:100]}...")
    
    # Split message into words (potential entities)
    words = message.content.split()
    current_phrase = []
    entities = []
    
    for i, word in enumerate(words):
        # Add word to current phrase
        current_phrase.append(word)
        phrase = " ".join(current_phrase)
        
        # Get embedding for current phrase
        phrase_embedding = await get_embedding(phrase)
        
        # Find best matching reference entity in this mondo
        ref_entity, similarity = await find_best_match(phrase_embedding, message.mondo)
        
        if similarity >= settings.ENTITY_CONFIDENCE_THRESHOLD and ref_entity:
            logger.info(f"Phrase '{phrase}' matches reference entity '{ref_entity.text}' with confidence {similarity:.3f}")
            
            # Calculate position in original text
            start_pos = message.content.find(phrase)
            end_pos = start_pos + len(phrase)
            
            # Create entity
            entity = await Entity.objects.acreate(
                text=phrase,
                type=ref_entity.type,
                confidence=similarity,
                start_position=start_pos,
                end_position=end_pos,
                message=message,
                reference_entity=ref_entity
            )
            
            # Store the embedding
            entity.set_embedding_array(phrase_embedding)
            await sync_to_async(entity.save)()
            
            entities.append(entity)
            
            # Reset current phrase
            current_phrase = []
        elif len(current_phrase) >= 3:  # Limit phrase length
            current_phrase.pop(0)  # Remove oldest word
    
    logger.info(f"Created {len(entities)} entities")
    return entities 
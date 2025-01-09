import re
from typing import List, Dict, Tuple, Callable, Awaitable, Union
import numpy as np
from django.conf import settings
from asgiref.sync import sync_to_async
from entities.models import Entity, EntityReference, EntityArchetype
import asyncio

EmbeddingFunction = Union[Callable[[str], np.ndarray], Callable[[str], Awaitable[np.ndarray]]]

class StreamingEntityDetector:
    def __init__(self, mondo_id: int, buffer_size: int = 5):
        self.mondo_id = mondo_id
        self.buffer_size = buffer_size
        self.context_buffer = []
        self.last_incomplete_word = ""
        self.archetype = None  # Will be set in tests
        
    def _get_context(self) -> str:
        """Get the current context window as a string."""
        return " ".join(self.context_buffer[-self.buffer_size:])
        
    def _update_context(self, text: str):
        """Update the context buffer with new text."""
        words = text.split()
        if words:
            self.context_buffer.extend(words)
            if len(self.context_buffer) > self.buffer_size:
                self.context_buffer = self.context_buffer[-self.buffer_size:]
                
    async def _find_pattern_matches(self, text: str) -> List[Dict]:
        """Find potential entity matches in text using pattern matching."""
        # Get all entity references for this mondo
        refs = await sync_to_async(list)(
            EntityReference.objects.filter(mondo_id=self.mondo_id).values_list('text', flat=True)
        )
        
        matches = []
        for ref in refs:
            # Look for word boundary matches (case-insensitive)
            pattern = r'\b' + re.escape(ref) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
                
        # Sort by position
        matches.sort(key=lambda x: x['start'])
        return matches
        
    async def _get_similar_references(self, text: str, embedding_fn: EmbeddingFunction) -> List[Dict]:
        """Find similar entity references using vector similarity."""
        # Handle both sync and async embedding functions
        if asyncio.iscoroutinefunction(embedding_fn):
            text_embedding = await embedding_fn(text)
        else:
            text_embedding = embedding_fn(text)
        
        # Get references with embeddings
        refs = await sync_to_async(list)(
            EntityReference.objects.filter(mondo_id=self.mondo_id).select_related('archetype').values('id', 'text', 'embedding', 'archetype__name')
        )
        
        similar_refs = []
        for ref in refs:
            # If text matches exactly, return with high confidence
            if text.lower() == ref['text'].lower():
                similar_refs.append({
                    'id': ref['id'],
                    'text': ref['text'],
                    'confidence': 1.0,
                    'archetype_name': ref['archetype__name'],
                    'embedding': ref['embedding']
                })
                continue
                
            # Skip if either embedding is all zeros
            if not np.any(text_embedding) or not np.any(ref['embedding']):
                continue
                
            # Calculate cosine similarity
            similarity = np.dot(text_embedding, ref['embedding']) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(ref['embedding'])
            )
            
            if similarity > settings.ENTITY_CONFIDENCE_THRESHOLD:
                similar_refs.append({
                    'id': ref['id'],
                    'text': ref['text'],
                    'confidence': float(similarity),
                    'archetype_name': ref['archetype__name'],
                    'embedding': ref['embedding']
                })
                
        return similar_refs
        
    async def process_chunk(self, text: str, message_id: int, embedding_fn: EmbeddingFunction) -> Tuple[str, List[Dict]]:
        """Process a chunk of text and detect entities."""
        # Handle incomplete words from previous chunk
        if self.last_incomplete_word:
            text = self.last_incomplete_word + text
            self.last_incomplete_word = ""
            
        # Check if chunk ends with incomplete word
        if not text.endswith(" "):
            words = text.split()
            if words:
                last_word = words[-1]
                last_word_start = text.rindex(last_word)
                if not text[last_word_start-1:last_word_start].isspace():
                    # Only treat as incomplete if it's not a complete word
                    self.last_incomplete_word = last_word
                    text = text[:last_word_start]
                
        # Update context
        self._update_context(text)
        
        # If we have only an incomplete word, return it as is
        if not text and self.last_incomplete_word:
            return self.last_incomplete_word, []
        
        # Find pattern matches
        matches = await self._find_pattern_matches(text)
        
        # Track processed spans to avoid overlaps
        processed_spans = set()
        entities = []
        marked_text = text
        
        # Sort matches by position to process them in order
        matches.sort(key=lambda x: x['start'])
        
        # Keep track of offset due to added markup
        offset = 0
        
        for match in matches:
            # Skip if span overlaps with processed
            span = (match['start'], match['end'])
            if any(start <= span[0] < end or start < span[1] <= end 
                  for start, end in processed_spans):
                continue
                
            # Get similar references
            similar_refs = await self._get_similar_references(
                match['text'],
                embedding_fn
            )
            
            if similar_refs:
                # Use most similar reference
                ref = similar_refs[0]
                
                # Get the reference entity
                reference = await sync_to_async(EntityReference.objects.select_related('archetype').get)(id=ref['id'])
                
                # Get embedding for the entity
                if asyncio.iscoroutinefunction(embedding_fn):
                    entity_embedding = await embedding_fn(match['text'])
                else:
                    entity_embedding = embedding_fn(match['text'])
                
                # Create entity
                entity = await sync_to_async(Entity.objects.create)(
                    text=match['text'],
                    archetype=self.archetype,  # Use the provided archetype
                    reference_entity=reference,
                    message_id=message_id,
                    start_position=match['start'],
                    end_position=match['end'],
                    confidence=ref['confidence'],
                    embedding=entity_embedding
                )
                
                # Add markup
                entity_tag = f'<entity id="{entity.id}">{match["text"]}</entity>'
                start_pos = match['start'] + offset
                end_pos = match['end'] + offset
                marked_text = (
                    marked_text[:start_pos] +
                    entity_tag +
                    marked_text[end_pos:]
                )
                
                # Update offset for next iteration
                offset += len(entity_tag) - len(match['text'])
                
                entities.append({
                    'id': entity.id,
                    'text': match['text'],
                    'type': ref['archetype_name'],  # Use the prefetched archetype name
                    'confidence': ref['confidence'],
                    'start': match['start'],
                    'end': match['end']
                })
                
                processed_spans.add(span)
                
        return marked_text, entities 
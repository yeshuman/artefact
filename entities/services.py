import re
from typing import List, Dict, Tuple, Callable
import numpy as np
from django.conf import settings
from asgiref.sync import sync_to_async
from entities.models import Entity, EntityReference, EntityArchetype

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
            # Look for word boundary matches
            pattern = r'\b' + re.escape(ref) + r'\b'
            for match in re.finditer(pattern, text):
                matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
                
        # Sort by position
        matches.sort(key=lambda x: x['start'])
        return matches
        
    async def _get_similar_references(self, text: str, embedding_fn: Callable) -> List[Dict]:
        """Find similar entity references using vector similarity."""
        text_embedding = embedding_fn(text)
        
        # Get references with embeddings
        refs = await sync_to_async(list)(
            EntityReference.objects.filter(mondo_id=self.mondo_id).values_list('id', 'text', 'embedding')
        )
        
        similar_refs = []
        for ref_id, ref_text, ref_embedding in refs:
            # If text matches exactly, return with high confidence
            if text.lower() == ref_text.lower():
                similar_refs.append({
                    'id': ref_id,
                    'text': ref_text,
                    'confidence': 1.0
                })
                continue
                
            # Skip if either embedding is all zeros
            if not np.any(text_embedding) or not np.any(ref_embedding):
                continue
                
            # Calculate cosine similarity
            similarity = np.dot(text_embedding, ref_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(ref_embedding)
            )
            
            if similarity > settings.ENTITY_CONFIDENCE_THRESHOLD:
                similar_refs.append({
                    'id': ref_id,
                    'text': ref_text,
                    'confidence': float(similarity)
                })
                
        return similar_refs
        
    async def process_chunk(self, text: str, message_id: int, embedding_fn: Callable) -> Tuple[str, List[Dict]]:
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
                reference = await sync_to_async(EntityReference.objects.get)(id=ref['id'])
                
                # Create entity
                entity = await Entity.objects.acreate(
                    text=match['text'],
                    archetype=self.archetype,  # Use the provided archetype
                    reference_entity=reference,
                    message_id=message_id,
                    start_position=match['start'],
                    end_position=match['end'],
                    confidence=ref['confidence'],
                    embedding=embedding_fn(match['text'])
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
                    'type': reference.type,
                    'confidence': ref['confidence'],
                    'start': match['start'],
                    'end': match['end']
                })
                
                processed_spans.add(span)
                
        return marked_text, entities 
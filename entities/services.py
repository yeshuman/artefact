import re
from typing import List, Dict, Tuple, Callable, Awaitable, Union
import numpy as np
from django.conf import settings
from asgiref.sync import sync_to_async
from entities.models import Entity, EntityReference, EntityArchetype
import asyncio
import sys
import uuid
from sklearn.metrics.pairwise import cosine_similarity

EmbeddingFunction = Union[Callable[[str], np.ndarray], Callable[[str], Awaitable[np.ndarray]]]

class StreamingEntityDetector:
    def __init__(self, mondo_id: int):
        self.mondo_id = mondo_id
        self.archetype = None
        self.context = ""
        self.incomplete_word = ""
        self.context_size = 1000  # Keep last 1000 chars
        self.similarity_threshold = 0.7  # Lower threshold for better recall
        self.seen_words = set()  # Track processed words
        
    async def process_chunk(self, text: str, message_id: int, get_embedding_fn) -> Tuple[str, List[Dict]]:
        """Process a chunk of text and detect entities."""
        # Update context first
        self.context += text
        if len(self.context) > self.context_size:
            self.context = self.context[-self.context_size:]
            
        # Calculate context offset
        context_offset = len(self.context) - len(text)
        
        # Process the text in two parts if we have an incomplete word
        matches = []
        
        if self.incomplete_word:
            # First try to find entities that span the boundary
            boundary_text = self.incomplete_word + text.split()[0] if text.split() else self.incomplete_word
            boundary_matches = await self._find_entities(
                boundary_text,
                context_offset - len(self.incomplete_word),
                message_id,
                get_embedding_fn
            )
            matches.extend(boundary_matches)
        
        # Then process the rest of the text
        text_matches = await self._find_entities(
            text,
            context_offset,
            message_id,
            get_embedding_fn
        )
        matches.extend(text_matches)
        
        # Add markup for detected entities
        marked_text = text
        # Sort matches by position in reverse to avoid markup interference
        for match in sorted(matches, key=lambda x: x["start_position"], reverse=True):
            start_in_chunk = match["start_position"] - context_offset
            end_in_chunk = start_in_chunk + len(match["text"])
            if 0 <= start_in_chunk < len(text):
                marked_text = (
                    marked_text[:start_in_chunk] +
                    f'<entity id="{match["id"]}" type="{match["type"]}">{match["text"]}</entity>' +
                    marked_text[end_in_chunk:]
                )
        
        # Store detected entities
        for match in matches:
            await Entity.objects.acreate(
                text=match["text"],
                archetype=self.archetype,
                message_id=message_id,
                confidence=match["confidence"],
                start_position=match["start_position"],
                end_position=match["end_position"],
                reference_entity=match["reference_entity"],
                embedding=match["embedding"]  # Store the embedding used for the match
            )
        
        # Update incomplete word - capture any partial word at the end
        last_word_match = re.search(r'\w+$', text)
        self.incomplete_word = last_word_match.group() if last_word_match else ""
        
        return marked_text, matches
        
    async def _find_entities(
        self,
        text: str,
        offset: int,
        message_id: int,
        get_embedding_fn
    ) -> List[Dict]:
        """Find entities in a piece of text with the given offset."""
        matches = []
        # Get all reference entities for this archetype
        refs = [ref async for ref in EntityReference.objects.filter(archetype=self.archetype)]
        
        # Create a pattern that matches any of the reference entity texts
        # Allow for some variations in the text (e.g. "Louvre Museum" matches "Louvre")
        ref_texts = []
        for ref in refs:
            parts = ref.text.split()
            if len(parts) > 1:
                # Add both full name and main part
                ref_texts.append(re.escape(ref.text))
                ref_texts.append(re.escape(parts[0]))
            else:
                ref_texts.append(re.escape(ref.text))
        
        pattern = r'\b(?:' + '|'.join(ref_texts) + r')\b'
        words = re.finditer(pattern, text, re.IGNORECASE)
        
        for word_match in words:
            word = word_match.group()
            chunk_start = word_match.start()
            
            # Skip if we've already processed this word at this position
            word_key = f"{word}_{offset + chunk_start}"
            if word_key in self.seen_words:
                continue
            self.seen_words.add(word_key)
            
            # Get embedding for word
            embedding_result = get_embedding_fn(word)
            if asyncio.iscoroutine(embedding_result):
                embedding = await embedding_result
            else:
                embedding = embedding_result
            
            # Find the matching reference entity
            # Match either exact text or first word
            ref = next(
                r for r in refs 
                if r.text.lower() == word.lower() or r.text.lower().startswith(word.lower())
            )
            
            # Create the match with exact match confidence
            match = {
                "text": word,
                "type": self.archetype.name,
                "confidence": 1.0 if ref.text.lower() == word.lower() else 0.9,
                "id": str(uuid.uuid4()),
                "start_position": offset + chunk_start,
                "end_position": offset + chunk_start + len(word),
                "reference_entity": ref,
                "embedding": embedding
            }
            matches.append(match)
                
        return matches
        
    def _get_context(self) -> str:
        """Get the current context window."""
        return self.context 
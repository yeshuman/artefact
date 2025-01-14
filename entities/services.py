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
        self.similarity_threshold = 0.60  # Lower base threshold for archetype matching
        self.reference_similarity_threshold = 0.65  # Lower base threshold for reference matching
        self.exact_match_bonus = 0.15  # Increased bonus for exact text matches
        self.seen_words = set()  # Track processed words
        self.reference_confidence_threshold = 0.95  # Threshold for creating new reference entities
        self.min_references_for_bonus = 2  # Reduced minimum references needed for bonus
        self.max_references_for_bonus = 10  # Maximum references considered for bonus
        self.reference_bonus_factor = 0.05  # Increased bonus factor per reference
        self.archetype_weight = 0.4  # Weight for archetype similarity
        self.reference_weight = 0.6  # Weight for reference similarity
        
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
        
        # Filter out matches with low confidence
        matches = [m for m in matches if m["confidence"] >= self.reference_similarity_threshold]
        
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
        
        # Store detected entities and potentially create new reference entities
        stored_matches = []
        for match in matches:
            try:
                # Store the detected entity
                entity = await Entity.objects.acreate(
                    text=match["text"],
                    archetype=self.archetype,
                    message_id=message_id,
                    confidence=match["confidence"],
                    start_position=match["start_position"],
                    end_position=match["end_position"],
                    reference_entity=match["reference_entity"],
                    embedding=match["embedding"]  # Store the embedding used for the match
                )
                stored_matches.append(match)
                
                # If this is a high confidence match and not already a reference,
                # store it as a new reference entity
                if match["confidence"] >= self.reference_confidence_threshold:
                    await self._store_as_reference(match)
            except Exception as e:
                print(f"Error storing entity: {e}", file=sys.stderr)
                continue
        
        # Update incomplete word - capture any partial word at the end
        last_word_match = re.search(r'\w+$', text)
        self.incomplete_word = last_word_match.group() if last_word_match else ""
        
        return marked_text, stored_matches
        
    async def _store_as_reference(self, match: Dict) -> None:
        """Store a high-confidence match as a reference entity if it doesn't exist."""
        # Check if this text is already a reference entity for this archetype
        existing = await EntityReference.objects.filter(
            text=match["text"],
            archetype=self.archetype,
            mondo_id=self.mondo_id
        ).afirst()
        
        if not existing:
            # Create new reference entity
            await EntityReference.objects.acreate(
                text=match["text"],
                archetype=self.archetype,
                mondo_id=self.mondo_id,
                embedding=match["embedding"]
            )
        
    async def _calculate_confidence(self, ref_similarity: float, archetype_similarity: float, num_references: int) -> float:
        """Calculate final confidence score taking into account archetype similarity, reference similarity, and number of references."""
        # Increase weight for reference similarity to give more importance to known entities
        self.archetype_weight = 0.35  # Slightly reduce archetype weight
        self.reference_weight = 0.65  # Increase reference weight
        
        # Calculate weighted similarity with increased emphasis on reference matching
        weighted_similarity = (self.archetype_weight * archetype_similarity + 
                             self.reference_weight * ref_similarity)
        
        # Add bonus based on number of references if we have more than minimum
        if num_references >= self.min_references_for_bonus:
            # Calculate bonus factor based on number of references
            bonus_references = min(num_references - self.min_references_for_bonus, 
                               self.max_references_for_bonus - self.min_references_for_bonus)
            # Increase reference bonus factor for stronger differentiation
            self.reference_bonus_factor = 0.08  # Increased from 0.05
            reference_bonus = bonus_references * self.reference_bonus_factor
            weighted_similarity = weighted_similarity * (1.0 + reference_bonus)
        
        # Add a small bonus for very high archetype similarity
        if archetype_similarity > 0.8:
            archetype_bonus = 0.1 * (archetype_similarity - 0.8)  # Up to 0.1 bonus for perfect match
            weighted_similarity += archetype_bonus
        
        return min(weighted_similarity, 1.0)  # Cap at 1.0

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
        num_references = len(refs)
        
        # Create patterns for multi-word and single-word matches
        multi_word_refs = [ref for ref in refs if len(ref.text.split()) > 1]
        single_word_refs = [ref for ref in refs if len(ref.text.split()) == 1]
        
        # First try to match multi-word references with a lower initial threshold
        self.similarity_threshold = 0.55  # Lower threshold for initial matching
        
        for ref in multi_word_refs:
            pattern = re.compile(rf'\b{re.escape(ref.text)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                word = match.group()
                chunk_start = match.start()
                
                # Skip if we've already processed this word at this position
                word_key = f"{word}_{offset + chunk_start}"
                if word_key in self.seen_words:
                    continue
                
                # Get embedding for word
                embedding_result = get_embedding_fn(word)
                if asyncio.iscoroutine(embedding_result):
                    embedding = await embedding_result
                else:
                    embedding = embedding_result
                
                # Calculate similarity with archetype
                archetype_embedding = np.array(self.archetype.embedding, dtype=np.float32)
                archetype_similarity = cosine_similarity(
                    [embedding],
                    [archetype_embedding]
                )[0][0]
                
                # Calculate similarity with reference
                ref_embedding = np.array(ref.embedding, dtype=np.float32)
                ref_similarity = cosine_similarity([embedding], [ref_embedding])[0][0]
                
                # Apply exact match bonus
                if word.lower() == ref.text.lower():
                    ref_similarity += self.exact_match_bonus
                
                # Calculate final confidence with reference bonus
                confidence = await self._calculate_confidence(ref_similarity, archetype_similarity, num_references)
                
                # If word is similar enough to both archetype and reference, create match
                if archetype_similarity >= self.similarity_threshold and confidence >= self.reference_similarity_threshold:
                    # Create match
                    match = {
                        "text": word,
                        "type": self.archetype.name,
                        "confidence": confidence,
                        "id": str(uuid.uuid4()),
                        "start_position": offset + chunk_start,
                        "end_position": offset + chunk_start + len(word),
                        "reference_entity": ref,
                        "embedding": embedding
                    }
                    matches.append(match)
                    self.seen_words.add(word_key)
        
        # Then try to match phrases in text with context-aware thresholds
        words = text.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 5, len(words) + 1)):  # Look at phrases up to 4 words
                phrase = " ".join(words[i:j])
                chunk_start = text.find(phrase, text.find(" ".join(words[:i])) if i > 0 else 0)
                
                # Skip if we've already processed this phrase at this position
                word_key = f"{phrase}_{offset + chunk_start}"
                if word_key in self.seen_words:
                    continue
                
                # Get embedding for phrase
                embedding_result = get_embedding_fn(phrase)
                if asyncio.iscoroutine(embedding_result):
                    embedding = await embedding_result
                else:
                    embedding = embedding_result
                
                # Calculate similarity with archetype
                archetype_embedding = np.array(self.archetype.embedding, dtype=np.float32)
                archetype_similarity = cosine_similarity(
                    [embedding],
                    [archetype_embedding]
                )[0][0]
                
                # Adjust threshold based on phrase length and context
                phrase_threshold = max(
                    self.similarity_threshold - (0.05 * (len(phrase.split()) - 1)),  # Lower threshold for longer phrases
                    0.45  # But not lower than this
                )
                
                # If phrase is similar enough to archetype, check reference entities
                if archetype_similarity >= phrase_threshold:
                    # Find best matching reference if any
                    best_ref = None
                    best_similarity = 0
                    
                    for ref in refs:
                        ref_embedding = np.array(ref.embedding, dtype=np.float32)
                        similarity = cosine_similarity([embedding], [ref_embedding])[0][0]
                        
                        # Apply exact match bonus
                        if phrase.lower() == ref.text.lower():
                            similarity += self.exact_match_bonus
                        # Apply partial match bonus for overlapping terms
                        elif any(word.lower() in ref.text.lower() or ref.text.lower() in word.lower() 
                               for word in phrase.split()):
                            similarity += self.exact_match_bonus * 0.5
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_ref = ref
                    
                    if best_ref:
                        # Calculate final confidence with reference bonus
                        confidence = await self._calculate_confidence(best_similarity, archetype_similarity, num_references)
                        
                        # Lower confidence threshold for longer phrases
                        phrase_confidence_threshold = max(
                            self.reference_similarity_threshold - (0.05 * (len(phrase.split()) - 1)),
                            0.55  # But not lower than this
                        )
                        
                        if confidence >= phrase_confidence_threshold:
                            # Create match
                            match = {
                                "text": phrase,
                                "type": self.archetype.name,
                                "confidence": confidence,
                                "id": str(uuid.uuid4()),
                                "start_position": offset + chunk_start,
                                "end_position": offset + chunk_start + len(phrase),
                                "reference_entity": best_ref,
                                "embedding": embedding
                            }
                            matches.append(match)
                            self.seen_words.add(word_key)
        
        # Finally try to match single words with original thresholds
        self.similarity_threshold = 0.60  # Reset to original threshold
        
        for word_match in re.finditer(r'\b\w+\b', text):
            word = word_match.group()
            chunk_start = word_match.start()
            
            # Skip if we've already processed this word at this position
            word_key = f"{word}_{offset + chunk_start}"
            if word_key in self.seen_words:
                continue
            
            # Get embedding for word
            embedding_result = get_embedding_fn(word)
            if asyncio.iscoroutine(embedding_result):
                embedding = await embedding_result
            else:
                embedding = embedding_result
            
            # Calculate similarity with archetype
            archetype_embedding = np.array(self.archetype.embedding, dtype=np.float32)
            archetype_similarity = cosine_similarity(
                [embedding],
                [archetype_embedding]
            )[0][0]
            
            # If word is similar enough to archetype, check reference entities
            if archetype_similarity >= self.similarity_threshold:
                # Find best matching reference if any
                best_ref = None
                best_similarity = 0
                
                for ref in refs:
                    ref_embedding = np.array(ref.embedding, dtype=np.float32)
                    similarity = cosine_similarity([embedding], [ref_embedding])[0][0]
                    
                    # Apply exact match bonus
                    if word.lower() == ref.text.lower():
                        similarity += self.exact_match_bonus
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_ref = ref
                
                if best_ref:
                    # Calculate final confidence with reference bonus
                    confidence = await self._calculate_confidence(best_similarity, archetype_similarity, num_references)
                    
                    if confidence >= self.reference_similarity_threshold:
                        # Create match
                        match = {
                            "text": word,
                            "type": self.archetype.name,
                            "confidence": confidence,
                            "id": str(uuid.uuid4()),
                            "start_position": offset + chunk_start,
                            "end_position": offset + chunk_start + len(word),
                            "reference_entity": best_ref,
                            "embedding": embedding
                        }
                        matches.append(match)
                        self.seen_words.add(word_key)
        
        return matches
        
    def _get_context(self) -> str:
        """Get the current context window."""
        return self.context 
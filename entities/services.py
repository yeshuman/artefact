import re
from typing import List, Dict, Tuple, Callable, Awaitable, Union
import numpy as np
from django.conf import settings
from asgiref.sync import sync_to_async
from entities.models import Entity, EntityReference, EntityArchetype
from mondos.models import Message
import asyncio
import sys
import uuid
from scipy import spatial
from utils.embeddings import get_embedding

EmbeddingFunction = Union[Callable[[str], np.ndarray], Callable[[str], Awaitable[np.ndarray]]]

class StreamingEntityDetector:
    """Detects entities in streaming text using embeddings."""
    
    CONTEXT_WINDOW_SIZE = 10
    CONFIDENCE_THRESHOLD = 0.85  # Lower threshold to detect more entities
    
    def __init__(self, message_id: int):
        """Initialize the detector with a message ID."""
        self.message_id = message_id
        self.context_window = []
        self.detected_spans = set()
        self.archetype = None

    def reset(self):
        """Reset the detector state."""
        self.context_window = []
        self.detected_spans = set()
        self.archetype = None

    async def process_chunk(self, chunk: str, message_id: int, embedding_fn: Callable[[str], Awaitable[np.ndarray]]) -> Tuple[str, List[Dict]]:
        """Process a chunk of text and detect entities.
        
        Args:
            chunk: The text chunk to process
            message_id: The ID of the message being processed
            embedding_fn: Function to get embeddings for text
            
        Returns:
            Tuple of (marked up text, list of detected entities)
        """
        # Split chunk into words and add to context window
        words = chunk.split()
        self.context_window.extend(words)
        
        # Keep context window at fixed size
        if len(self.context_window) > self.CONTEXT_WINDOW_SIZE:
            self.context_window = self.context_window[-self.CONTEXT_WINDOW_SIZE:]

        marked_text = chunk
        detected_entities = []

        # Get reference entities for comparison
        reference_entities = await sync_to_async(list)(EntityReference.objects.filter(archetype=self.archetype))

        # Look for potential entity spans in context window
        for i in range(len(self.context_window)):
            for j in range(i + 1, len(self.context_window) + 1):
                span = " ".join(self.context_window[i:j])
                
                # Skip if already detected
                if span in self.detected_spans:
                    continue

                try:
                    # Get embedding for span
                    span_embedding = await embedding_fn(span)
                    span_embedding = span_embedding.reshape(1, -1)

                    # Compare against reference entities
                    for reference_entity in reference_entities:
                        ref_embedding = np.array(reference_entity.embedding).reshape(1, -1)
                        cosine_similarity = float(1 - spatial.distance.cosine(span_embedding.flatten(), ref_embedding.flatten()))

                        # Create entity if confidence is high enough
                        if cosine_similarity >= self.CONFIDENCE_THRESHOLD:
                            start_idx = i
                            end_idx = j - 1

                            entity = await Entity.objects.acreate(
                                text=span,
                                archetype=self.archetype,
                                confidence=float(cosine_similarity),
                                start_position=start_idx,
                                end_position=end_idx,
                                message=await Message.objects.aget(id=message_id),
                                reference_entity=reference_entity,
                                embedding=span_embedding.tolist()
                            )

                            # Create entity dict for response
                            entity_dict = {
                                "id": entity.id,
                                "text": span,
                                "type": self.archetype.name,
                                "confidence": cosine_similarity,
                                "start": start_idx,
                                "end": end_idx
                            }
                            detected_entities.append(entity_dict)
                            self.detected_spans.add(span)

                            # Add markup
                            marked_text = re.sub(
                                r'\b' + re.escape(span) + r'\b',
                                f'<entity id="{entity.id}" type="{self.archetype.name}">{span}</entity>',
                                marked_text
                            )
                            break

                except Exception as e:
                    print(f"Error processing span {span}: {e}", file=sys.stderr)
                    continue

        return marked_text, detected_entities
        
    def _get_context(self) -> str:
        """Get the current context window."""
        return self.context 
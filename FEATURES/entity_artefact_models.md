# Entity and Artefact Models

## Overview
Defines the core models for handling travel-related entities detected in conversations
and their transformation into enriched artefacts.

## Models

### 1. Entity Model
```python
class Entity(models.Model):
    """Raw travel-related entity detected in conversation text."""
    
    # Core Fields
    text = models.CharField(max_length=255)
    type = models.CharField(max_length=50, choices=[
        ('PLACE', 'Place'),
        ('SITE', 'Cultural Site'),
        ('LANDMARK', 'Landmark'),
        ('REGION', 'Region'),
        ('EVENT', 'Historical Event')
    ])
    confidence = models.FloatField()
    
    # Detection Context
    message = models.ForeignKey('mondos.Message', on_delete=models.CASCADE)
    start_position = models.IntegerField()
    end_position = models.IntegerField()
    detected_at = models.DateTimeField(auto_now_add=True)
    
    # Vector Embedding
    embedding = models.JSONField(null=True)
    
    class Meta:
        verbose_name_plural = "Entities"
        indexes = [
            models.Index(fields=['type', 'confidence']),
            models.Index(fields=['message', 'start_position'])
        ]
```

### 2. Artefact Model
```python
class Artefact(models.Model):
    """Enriched travel-related object derived from detected entities."""
    
    # Core Fields
    name = models.CharField(max_length=255)
    slug = models.SlugField(unique=True)
    description = models.TextField(null=True)
    
    # State Management
    state = models.CharField(max_length=20, choices=[
        ('IDENTIFIED', 'Entity Detected'),
        ('LOADING', 'API Enrichment In Progress'),
        ('LOADED', 'Data Available'),
        ('FAILED', 'Enrichment Failed')
    ])
    state_changed_at = models.DateTimeField(auto_now=True)
    
    # Relationships
    source_entity = models.ForeignKey(Entity, on_delete=models.SET_NULL, null=True)
    related_artefacts = models.ManyToManyField('self', symmetrical=True)
    
    # API Data
    google_place_id = models.CharField(max_length=255, null=True)
    google_place_data = models.JSONField(null=True)
    tripadvisor_data = models.JSONField(null=True)
    
    # Metadata
    location = models.JSONField(null=True)  # {lat: float, lng: float}
    tags = ArrayField(models.CharField(max_length=50), default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['state', 'state_changed_at']),
            models.Index(fields=['google_place_id'])
        ]
```

### 3. ArtefactEnrichmentTask Model
```python
class ArtefactEnrichmentTask(models.Model):
    """Background task for enriching artefacts with API data."""
    
    artefact = models.ForeignKey(Artefact, on_delete=models.CASCADE)
    source = models.CharField(max_length=50, choices=[
        ('GOOGLE_PLACES', 'Google Places API'),
        ('TRIPADVISOR', 'TripAdvisor API')
    ])
    state = models.CharField(max_length=20, choices=[
        ('PENDING', 'Task Pending'),
        ('IN_PROGRESS', 'Task Running'),
        ('COMPLETED', 'Task Completed'),
        ('FAILED', 'Task Failed')
    ])
    error_message = models.TextField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['state', 'created_at']),
            models.Index(fields=['artefact', 'source'])
        ]
```

## Processing Pipeline

### 1. Entity Detection
```python
async def detect_entities(message: Message) -> list[Entity]:
    """Detect potential travel-related entities in message content."""
    # 1. Prepare content for analysis
    content = message.content
    
    # 2. Run vector similarity search
    matches = await vector_search(content)
    
    # 3. Validate and classify matches
    entities = []
    for match in matches:
        if match.confidence > settings.ENTITY_CONFIDENCE_THRESHOLD:
            entity = await Entity.objects.acreate(
                text=match.text,
                type=match.type,
                confidence=match.confidence,
                message=message,
                start_position=match.start,
                end_position=match.end
            )
            entities.append(entity)
    
    return entities
```

### 2. Entity to Artefact Transformation
```python
async def transform_entity(entity: Entity) -> Artefact:
    """Transform a detected entity into an artefact."""
    # 1. Create or get artefact
    artefact = await Artefact.objects.filter(
        source_entity=entity
    ).afirst()
    
    if not artefact:
        artefact = await Artefact.objects.acreate(
            name=entity.text,
            slug=slugify(entity.text),
            state='IDENTIFIED',
            source_entity=entity
        )
    
    # 2. Schedule enrichment tasks
    await ArtefactEnrichmentTask.objects.acreate(
        artefact=artefact,
        source='GOOGLE_PLACES',
        state='PENDING'
    )
    
    return artefact
```

### 3. Artefact Enrichment
```python
async def enrich_artefact(task: ArtefactEnrichmentTask) -> None:
    """Enrich artefact with external API data."""
    try:
        # 1. Update task state
        task.state = 'IN_PROGRESS'
        await sync_to_async(task.save)()
        
        # 2. Get API client
        client = get_api_client(task.source)
        
        # 3. Fetch and store data
        data = await client.fetch_place_details(task.artefact.name)
        
        # 4. Update artefact
        task.artefact.google_place_data = data
        task.artefact.state = 'LOADED'
        await sync_to_async(task.artefact.save)()
        
        # 5. Complete task
        task.state = 'COMPLETED'
        task.completed_at = timezone.now()
        await sync_to_async(task.save)()
        
    except Exception as e:
        # Handle failure
        task.state = 'FAILED'
        task.error_message = str(e)
        await sync_to_async(task.save)()
        
        task.artefact.state = 'FAILED'
        await sync_to_async(task.artefact.save)()
```

## Testing Strategy

### 1. Entity Detection Tests
- Vector similarity accuracy
- Entity classification
- Confidence scoring
- Position tracking
- Duplicate handling

### 2. Artefact Transformation Tests
- Entity to artefact mapping
- State transitions
- Slug generation
- Relationship management
- Task scheduling

### 3. API Integration Tests
- Rate limiting
- Error handling
- Data transformation
- State management
- Task queuing

### 4. Performance Tests
- Detection latency
- Enrichment timing
- API response times
- Database query optimization
- Vector search efficiency 

## Real-Time Processing Strategy

### 1. Stream Interception Pipeline
```python
async def process_stream_chunk(chunk: str) -> tuple[str, list[Entity]]:
    """Process each chunk of the LLM stream for entities in real-time."""
    
    # 1. Accumulate text for context
    buffer.append(chunk)
    current_text = ''.join(buffer[-5:])  # Keep last 5 chunks for context
    
    # 2. Quick pattern matching for potential entities
    potential_matches = await quick_pattern_match(current_text)
    if not potential_matches:
        return chunk, []
    
    # 3. Run lightweight vector similarity
    entities = []
    for match in potential_matches:
        confidence = await vector_similarity(match.text)
        if confidence > settings.REALTIME_CONFIDENCE_THRESHOLD:
            entity = await Entity.objects.acreate(
                text=match.text,
                confidence=confidence,
                # ... other fields
            )
            entities.append(entity)
    
    # 4. Enrich in background
    if entities:
        asyncio.create_task(enrich_entities(entities))
    
    # 5. Wrap detected entities in markup
    marked_chunk = await wrap_entities_in_chunk(chunk, entities)
    return marked_chunk, entities
```

### 2. Integration with Mondo Stream
```python
async def mondo_stream_generator():
    """Generate SSE events for the Mondo stream with real-time entity detection."""
    try:
        while True:
            if latest_message:
                stream = await llm_client.chat.completions.create(
                    messages=[...],
                    stream=True
                )
                
                async for chunk in stream:
                    if content := chunk.choices[0].delta.content:
                        # Process chunk for entities
                        marked_content, entities = await process_stream_chunk(content)
                        
                        # Emit content event
                        yield f'event: content\ndata: {marked_content}\n\n'
                        
                        # If entities found, emit artefact events
                        for entity in entities:
                            yield f'event: entity_detected\ndata: {json.dumps({"id": entity.id, "text": entity.text})}\n\n'
                            
                            # Start enrichment in background
                            asyncio.create_task(begin_enrichment(entity))
                    
            await asyncio.sleep(0.1)
            
    except asyncio.CancelledError:
        logger.info("Stream cancelled")
```

### 3. Background Enrichment
```python
async def begin_enrichment(entity: Entity):
    """Begin the enrichment process for a detected entity."""
    try:
        # 1. Transform to artefact
        artefact = await transform_entity(entity)
        
        # 2. Emit artefact creation event
        await emit_sse_event('artefact_created', {
            'id': artefact.id,
            'name': artefact.name,
            'state': 'IDENTIFIED'
        })
        
        # 3. Start API enrichment tasks
        tasks = []
        for api in ['GOOGLE_PLACES', 'TRIPADVISOR']:
            task = await ArtefactEnrichmentTask.objects.acreate(
                artefact=artefact,
                source=api,
                state='PENDING'
            )
            tasks.append(asyncio.create_task(enrich_artefact(task)))
        
        # 4. Wait for first successful enrichment
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks if one succeeds
        for task in pending:
            task.cancel()
            
    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        await emit_sse_event('artefact_error', {
            'entity_id': entity.id,
            'error': str(e)
        })
```

### 4. Client-Side Integration
```html
<div id="mondo-stream" 
     hx-sse="connect:/stream/mondo"
     sse-swap="content"
     hx-swap="beforeend">
</div>

<div id="artefact-status"
     hx-sse="connect:/stream/mondo"
     sse-swap="entity_detected,artefact_created,artefact_error">
</div>

<script>
htmx.on("sse:entity_detected", (evt) => {
    const entity = JSON.parse(evt.detail.data);
    highlightEntity(entity.id);
});

htmx.on("sse:artefact_created", (evt) => {
    const artefact = JSON.parse(evt.detail.data);
    updateArtefactPanel(artefact);
});
</script>
```

### Real-Time Processing Considerations

1. **Performance Optimization**
   - Quick pattern matching before expensive vector operations
   - Background processing for API enrichment
   - Cancellation of redundant API calls
   - Chunk buffering for context

2. **Stream Management**
   - Real-time entity markup injection
   - Parallel SSE event streams
   - Progressive UI updates
   - Error recovery

3. **Resource Management**
   - Task cancellation on success
   - Rate limiting for API calls
   - Connection pooling
   - Memory-efficient buffering

4. **User Experience**
   - Immediate entity highlighting
   - Progressive artefact loading
   - Graceful error handling
   - Responsive UI updates

## Entity Markup Strategy

### 1. Progressive Enhancement States
```html
<!-- Stage 1: Initial Detection (Potential Entity) -->
<span class="potential-entity" data-status="detecting">
    Kyoto Gardens
</span>

<!-- Stage 2: Entity Confirmed, Enrichment Pending -->
<span class="entity pending" 
      data-entity-id="123" 
      data-status="enriching"
      aria-disabled="true">
    Kyoto Gardens
    <span class="loading-indicator">⟳</span>
</span>

<!-- Stage 3: Enrichment Complete (Clickable Artefact) -->
<span class="entity active" 
      data-entity-id="123"
      data-artefact-id="456"
      data-status="ready"
      role="button"
      tabindex="0"
      onclick="focusArtefact('456')">
    Kyoto Gardens
    <span class="indicator">↗</span>
</span>
```

### 2. Styling and Visual States
```css
/* Base entity styles */
.potential-entity {
    border-bottom: 1px dashed #ccc;
    color: #666;
    cursor: default;
    transition: all 0.2s ease;
}

.entity.pending {
    border-bottom: 1px dashed #0066cc;
    color: #0066cc;
    opacity: 0.7;
    cursor: wait;
}

.entity.active {
    border-bottom: 2px solid #0066cc;
    color: #0066cc;
    cursor: pointer;
}

.entity.active:hover {
    background: #f0f7ff;
}

/* Loading indicator animation */
.loading-indicator {
    display: inline-block;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
```

### 3. State Management in Stream Processing
```python
async def process_stream_chunk(chunk: str) -> tuple[str, list[Entity]]:
    """Process chunk and wrap potential entities with appropriate markup."""
    potential_matches = await quick_pattern_match(chunk)
    
    for match in potential_matches:
        # Stage 1: Initially wrap as potential entity
        chunk = chunk.replace(
            match.text,
            f'<span class="potential-entity" data-status="detecting">{match.text}</span>'
        )
        
        # Quick vector similarity check
        if await is_valid_entity(match.text):
            # Stage 2: Update to pending entity
            entity = await Entity.objects.acreate(
                text=match.text,
                confidence=match.confidence,
                # ... other fields
            )
            
            chunk = chunk.replace(
                'potential-entity',
                f'entity pending" data-entity-id="{entity.id}'
            )
            
            # Begin enrichment process in background
            asyncio.create_task(begin_enrichment(entity))
    
    return chunk
```

### 4. Client-Side State Transitions
```javascript
// Handle entity state transitions via SSE
htmx.on("sse:entity_enriched", (evt) => {
    const {entity_id, artefact_id} = JSON.parse(evt.detail.data);
    const element = document.querySelector(`[data-entity-id="${entity_id}"]`);
    
    // Stage 3: Transform to active artefact
    element.classList.replace('pending', 'active');
    element.setAttribute('data-artefact-id', artefact_id);
    element.setAttribute('data-status', 'ready');
    element.removeAttribute('aria-disabled');
    
    // Update visual indicator
    element.querySelector('.loading-indicator')
        .replaceWith(createActiveIndicator());
});

// Handle enrichment failures
htmx.on("sse:entity_enrichment_failed", (evt) => {
    const {entity_id, error} = JSON.parse(evt.detail.data);
    const element = document.querySelector(`[data-entity-id="${entity_id}"]`);
    
    element.classList.replace('pending', 'failed');
    element.setAttribute('data-status', 'failed');
    element.setAttribute('title', `Enrichment failed: ${error}`);
});
```

### 5. Accessibility Considerations
- Use of `role="button"` for clickable entities
- `tabindex="0"` for keyboard navigation
- `aria-disabled="true"` during loading
- Clear visual state indicators
- Hover and focus states
- Error state feedback

### 6. Performance Optimizations
- Immediate visual feedback for potential matches
- Asynchronous state transitions
- CSS transitions for smooth state changes
- Efficient DOM updates via targeted selectors
- Debounced event handlers
- Cached selector lookups

### 7. Error States and Recovery
```html
<!-- Error State -->
<span class="entity failed" 
      data-entity-id="123"
      data-status="failed"
      title="Enrichment failed: API timeout">
    Kyoto Gardens
    <span class="error-indicator">!</span>
</span>
```

```css
.entity.failed {
    border-bottom: 1px dashed #cc0000;
    color: #cc0000;
    cursor: not-allowed;
}

.error-indicator {
    color: #cc0000;
    font-weight: bold;
}
```
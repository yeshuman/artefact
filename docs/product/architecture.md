# System Architecture

## Design Patterns
1. Streaming Architecture
   - Parallel mondo and artefact streams
   - Real-time entity detection
   - SSE for live updates
   - HTMX for dynamic UI

2. Processing Pipeline
   - Streaming text processing with context windows
   - Real-time entity detection during chat
   - Async artefact enrichment
   - Background task processing

3. Personality Management
   - Dynamic character system
   - System prompt templates
   - A/B testing framework
   - Personality evolution system

## Component Interactions
1. Frontend
   - Split-panel layout (chat | artefact details)
   - HTMX-enhanced templates
   - SSE event handling
   - Progressive enhancement

2. Backend
   - Async Django views
   - Streaming response handlers
   - Background task processors
   - API integration services

3. Storage
   - PostgreSQL for primary data
   - Redis for session management
   - Vector storage for embeddings
   - Cache for API responses

## Data Flow
1. Conversation Flow
   - User input → LLM processing
   - Streaming response → Entity detection
   - Entity → Artefact transformation
   - Artefact → API enrichment

2. Entity Processing
   - Text chunk → Pattern matching
   - Pattern → Vector similarity
   - Match → Entity creation
   - Entity → Artefact enrichment

3. State Management
   - Session tracking
   - Conversation context
   - Personality evolution
   - Artefact status updates 
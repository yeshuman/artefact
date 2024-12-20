# Project Specifications

## Current Features

### Chat System
- Real-time chat interface using Server-Sent Events (SSE)
- OpenAI GPT integration for AI responses
- Async message processing
- Message persistence in database
- Heartbeat mechanism for connection maintenance

## Planned Features

### Artefact Detection and Enrichment System

#### Overview
A real-time system for identifying travel-related entities from LLM conversations and transforming them into enriched Artefacts. The system processes streaming LLM responses to identify potential travel-related entities, converts them into Artefacts enriched with data from external APIs, and presents them in an interactive split-panel interface.

#### Core Components

1. Entity Detection Engine
   - Real-time stream analysis with performance monitoring
   - Entity identification through vector similarity search
   - Entity-to-Artefact transformation pipeline
   - Initial focus on geographical entities (cities, venues, points of interest)
   - Extensible architecture for future entity types (e.g., historical events)
   - Performance metric collection and analysis
   - Fallback to background processing if needed (based on performance metrics)

2. Data Model
   - Entity class for raw detected items
   - Base Artefact class for enriched data:
     * Derived from detected entities
     * Enriched with API data
     * Maintains relationship to source entity
   - Initial Artefact types:
     * Cities
     * Points of Interest
     * Venues (hotels, restaurants)
   - Vector embedding storage for entity detection
   - Artefact state tracking:
     * Identified: Entity detected and mapped to Artefact
     * Loading: API enrichment in progress
     * Loaded: Artefact data available for display
     * Failed: Unable to fetch/process enrichment data

3. Entity-to-Artefact Pipeline
   - Entity detection in stream
   - Entity validation and classification
   - Artefact creation/lookup
   - External API integration for enrichment:
     * Google Places API (primary source for venues and POIs)
     * TripAdvisor API (supplementary data and reviews)
   - Async processing system with task scheduling
   - Rate limiting implementation:
     * Per-service rate limits
     * Queue management
     * Priority handling
   - Data transformation and summarization
   - Background task scheduler for:
     * API data fetching
     * Future data freshness updates
     * Vector similarity search optimization

4. User Interface
   - Split-panel layout:
     * Left panel: Chat interface with entity-highlighted streaming responses
     * Right panel: Artefact details view
   - Dynamic content updates via HTMX
   - Semantic markup for identified entities:
     * Span tags with corresponding Artefact state indicators
     * Visual loading states
     * Interactive elements for Artefact information
   - Real-time state management:
     * Visual indicators for Artefact states
     * Loading animations during API fetches
     * Error state handling and recovery

#### Technical Implementation

1. Processing Pipeline
   - Stream interceptor for real-time entity analysis
   - Entity detector with vector similarity search
   - Entity-to-Artefact transformer
   - API coordinator for Artefact enrichment
   - Background task scheduler for async operations
   - Performance monitoring system

2. Data Storage
   - Vector database for entity embeddings
   - Relational storage for Artefact data
   - Entity-Artefact relationship mapping
   - Task queue for background processing
   - State management storage

3. API Integration
   - Async HTTP clients (httpx)
   - Rate limit management per service
   - Error handling and recovery
   - Response transformation and storage
   - Fallback strategies for API failures

4. Frontend Integration
   - HTMX controllers for dynamic updates
   - SSE for real-time streaming
   - Semantic HTML for artefact markup
   - State management and UI updates
   - Minimal JavaScript approach

#### Testing Strategy

1. Simulation Testing
   - Mock travel conversations
   - LLM response patterns
   - Artefact detection verification
   - Performance measurement and analysis
   - State transition testing

2. Integration Testing
   - End-to-end pipeline verification
   - API interaction testing
   - Rate limit compliance
   - UI component behavior
   - Error handling scenarios

3. Performance Testing
   - Stream processing latency
   - Entity detection accuracy
   - API response times
   - Resource utilization
   - Vector search performance

#### Monitoring and Metrics

1. Performance Metrics
   - Stream processing latency
   - Entity detection speed
   - API response times
   - Background task queue length
   - Resource utilization

2. Quality Metrics
   - Entity detection accuracy
   - API success rates
   - Error frequencies
   - State transition success rates

#### Future Considerations
- Data freshness management system
- Historical events integration
- Caching strategy implementation
- Performance optimization based on collected metrics
- Extended artefact type hierarchy
- User feedback integration
- Custom artefact linking

#### Conversation Flow with Artefacts

The system implements a natural conversation flow where:
1. Satori's responses contain tagged artefact references
2. Ronin can naturally focus on interesting artefacts
3. The conversation flows through discovered places and experiences

See FEATURES/artefact_conversation_flow.md for detailed implementation specs.

## Technical Architecture

### Backend Components
1. Django Application
   - Async views for streaming
   - Django ORM for data persistence
   - ASGI server support

2. External Services
   - OpenAI API integration
   - Async HTTP client (httpx)

### Data Models
python
Current Models
class HumanMessage:
text: str
timestamp: datetime
# ... other fields to be determined

### API Endpoints
- `/chat/` - Main chat interface
- `/chat/post/` - Message submission endpoint
- `/chat/stream/` - SSE streaming endpoint

## Integration Points
1. OpenAI API
   - Model: gpt-3.5-turbo
   - Streaming responses
   - Error handling

2. Frontend Integration
   - HTMX for dynamic updates
   - SSE for real-time streaming
   - Minimal JavaScript

## Performance Considerations
- Async processing for scalability
- Connection management
- Resource utilization
- Database query optimization

## Security Requirements
- Input sanitization
- API key protection
- Rate limiting
- CSRF protection




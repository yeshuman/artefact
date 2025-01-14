# System Patterns

## Architecture Patterns

### Entity Detection System
1. **Streaming Processing**
   - Chunk-based text processing
   - Maintains context window
   - Handles incomplete words at boundaries

2. **Entity Classification**
   - Archetype-based classification
   - Reference entity matching
   - Embedding-based similarity

3. **Progressive Learning**
   - High-confidence matches become references
   - Dynamic threshold adjustment
   - Performance metrics tracking

## Key Technical Decisions

### Embedding Strategy
- Using OpenAI's text-embedding-ada-002 model
- Cosine similarity for matching
- Caching of embeddings in database

### Confidence Scoring
- Weighted combination of archetype and reference similarity
- Progressive bonuses for multiple references
- Context-aware adjustments for phrases

### Testing Approach
- Combination of mock and real API tests
- Focus on relative confidence improvements
- Comprehensive test cases for edge conditions

## Database Schema
- EntityArchetype: Base types for classification
- EntityReference: Known good examples
- Entity: Detected instances
- Configuration: Dynamic system parameters 
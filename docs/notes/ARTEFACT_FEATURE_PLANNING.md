# Artefact Feature Planning Discussion

## Initial Requirements Discussion

When the LLM responds, the text will stream in (already accomplished). We need to:

1. Build knowledge base of Artefacts
2. Store Artefacts in vector embeddings as we identify and collect them
3. Support multiple Artefact types:
   - Geographical points of interest
   - Towns, cities, places
   - Hotels, restaurants, venues
   - (Future) Historical events

## Implementation Considerations

### Real-time Processing vs Background Tasks
- Initial attempt at real-time processing
- Monitor performance metrics
- Fall back to background processing if needed
- No hard performance metrics initially - observe and measure

### Artefact States
- Identified
- Loading
- Loaded
- Failed

### Testing Approach
- Simulate travel-related discussions
- Test LLM response patterns
- Verify artefact detection system

### Future Considerations
- Data freshness management via background tasks
- Historical events integration
- Caching strategy development 
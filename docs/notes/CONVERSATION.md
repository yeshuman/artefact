# Conversation

I'll help you update the SPECIFICATIONS.md file. What feature would you like to add? Based on the current specifications, we can add new features under the "Planned Features" section or create a new section if needed. Please let me know what feature you'd like to document and I'll help format it according to the project's documentation standards.

The current main sections in SPECIFICATIONS.md are:
- Current Features
- Planned Features
- Technical Architecture
- Integration Points
- Performance Considerations
- Security Requirements

What would you like to add?

ask a few questions before writing the spec; I'll start with an outline.

when the LLM responds the text will stream in; as we've accomplished.
we don't yet have a knowledge base of what I'll call Artefacts
we will store Artefacts in vector embeddings as we identify and collect them
Artefacts will be of several types; perhaps by sub-classing; which will be geographical points of interest; towns, cities or places; hotels, restaurants and venues; or anything that a traveller might be able to put into an itinerary when scheduling a trip.  Artefacts might also be historical events perhaps related to geographies.
Once we identify an artefact we will access APIs such as Google Places or Tripadvisor to see if we can fetch real world information about the artefact; upon which, we will interpret, perhaps summarise with an LLM, and store in our DB for future access via a django view rendered by htmx

We _could_ ask the LLM to wrap artefacts in a html tag at the time of streaming; however:

+ we don't know ahead of time if that information will be available in the API lookups

Or, we could do a vector similarity search as we accumulate data; and then have an async controller class with a task scheduler to fetch and summarise data in the background; such that when the next user comes along and initiates the LLM to make a similar response we will have the data on hand to render the view

I'll ask some clarifying questions to help shape the specification:
Data Collection & Processing:
Should we process the LLM stream in real-time to identify potential Artefacts, or analyze the complete response after streaming?
Do we want to maintain a confidence score for Artefact identification?
Should we track when an Artefact was last mentioned/queried to help with data freshness?
Artefact Classification:
Would you like to start with a fixed set of Artefact types, or should it be extensible?
For geographical Artefacts, what level of hierarchy should we maintain? (e.g., Country > State > City > Venue)
Should historical events be tied to specific dates/periods?
API Integration:
Should we prioritize certain APIs for certain Artefact types? (e.g., Google Places for venues, Wikipedia for historical events)
Do we want to store the raw API responses alongside our summarized versions?
Should we implement API fallbacks if primary sources are unavailable?
Vector Search Implementation:
What should be our strategy for embedding generation? Embed the entire Artefact description or key attributes only?
Should we implement a relevance threshold for similarity searches?
Do we want to maintain separate vector spaces for different Artefact types?
Background Processing:
What priority system should we use for the async task scheduler?
Should we implement rate limiting per API?
Do we want to set up automatic refresh cycles for stored data?
User Experience:
How should we handle cases where Artefact data is still being fetched/processed?
Should we provide visual indicators for Artefacts with available additional information?
Do we want to allow users to manually trigger data refreshes?
Let me know your thoughts on these questions, and I'll help draft a comprehensive specification that aligns with the project's existing structure and methodology.


I imagined attempting to process artefacts in real-time; but anticipated monitoring the performance costs

If the performance cost was high, then we might run a background task to prepare that artefact such that the next user/LLM event that re-discovers that artefact might have it rendered more immediately

2.
I'm naive to vector similarity search at this point and was wanting to experiement; however I was assuming that if we had the heirachy you mentioned; say, City; if the LLM response mentioned Paris then we would say that we have an identified artefact and we should query the APIs in order to build our data/view for the given artefact.

perhaps we can leave historical events to a later iteration.

other considerations i'm not sure of; we should probably implement a rate limiting strategy

I imagined that each word or chunk of our streaming response might be wrapped in a span tag or equivalent semantic markup; we might be able to implement visual indicators to those tags if we identify them to be an artefact to let the user know that we're fetching data that they might subsequently be able to click on the artefact link; which we would have htmx render an artefact panel on the right hand side; whilst our chat window with llm streaming responses occupies the left.


Thanks for those clarifications. Let me summarize the key points and ask a few follow-up questions before drafting the spec:
Real-time Processing Strategy:
Start with real-time processing
Monitor performance metrics
Fall back to background processing if needed
Would you like to set specific performance thresholds for this decision?
Initial Artefact Focus:
Starting with geographical entities (cities, places, venues)
Hierarchical structure (City as primary initial focus)
Historical events planned for future iteration
Should we define a minimum confidence threshold for Artefact identification?
UI/UX Approach:
Split-panel design (chat left, artefact details right)
Inline markup of potential artefacts using semantic HTML
Visual loading indicators for identified artefacts
Would you like to consider different states for artefacts? (e.g., identified, loading, loaded, failed)
Rate Limiting:
For both API calls and vector similarity searches
Should we implement per-service rate limits or a global rate limit?
Do we want to prioritize certain API lookups over others?
Data Storage & Caching:
How long should we cache API responses?
Should we implement a data freshness strategy?
Do we want to track usage patterns to optimize background processing?
Once you provide feedback on these points, I can draft a specification that includes:
Feature overview
Technical implementation details
Performance considerations
Data flow architecture
UI/UX specifications
Rate limiting strategy
Would you like me to proceed with drafting the specification based on what we know, or would you prefer to clarify any of these points first?


lets' leave data freshness and caching to a later concern; I imagine that we might eventually have a background task that then occupies itself with maintaining data freshness

let's not set hard performance metrics just yet; we will just observe and measure

we will consider different states for artefacts, ie identified, loading etc.

as part of our test running; we might run a process where we simulate discussions between a simulated travel user and the responses from the LLM; to explore how the system works.



// ... existing sections ...

### Artefact Detection and Enrichment System
- Real-time identification of travel-related entities in LLM responses
- Dynamic enrichment with external API data
- Interactive UI elements for discovered artefacts

#### Core Components
1. Artefact Detection
   - Real-time processing of LLM stream
   - Initial focus on geographical entities
   - Vector similarity search for entity identification
   - Performance monitoring and metrics collection

2. Data Enrichment
   - External API integration (Google Places, TripAdvisor)
   - Async processing pipeline
   - Background task scheduling for subsequent data fetching
   - Rate limiting implementation

3. UI Integration
   - Split-panel interface (chat | artefact details)
   - Semantic markup for identified artefacts
   - Visual state indicators:
     * Identified: Entity recognized
     * Loading: API data being fetched
     * Loaded: Data available for viewing
     * Failed: Unable to fetch data
   - HTMX-powered artefact detail panel

#### Data Models




#### Testing Strategy
1. Simulation Testing
   - Mock travel-related conversations
   - LLM response simulation
   - Artefact detection verification
   - Performance measurement

2. Integration Testing
   - API interaction testing
   - Rate limit verification
   - State transition testing
   - UI component testing

#### Future Considerations
- Data freshness management
- Historical event integration
- Caching strategy
- Performance optimization based on metrics
- Extended artefact type hierarchy

#### Performance Monitoring
- Stream processing latency
- Entity detection accuracy
- API response times
- Background task queue metrics
- Vector search performance

#### API Integration Points
- Google Places API
- TripAdvisor API
- Vector similarity search service
- Rate limiting service
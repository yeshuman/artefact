# Technical Context

## Technology Stack
1. Core Framework
   - Django 5.0+ for backend
   - HTMX for frontend interactions
   - Server-Sent Events for streaming
   - Redis for session management

2. Data Processing
   - OpenAI API for embeddings
   - numpy for numerical operations
   - scikit-learn for similarity calculations
   - PostgreSQL for data storage

3. External APIs
   - Google Places API
   - TripAdvisor API
   - OpenAI API
   - Vector similarity services

## Development Environment
- Async-first development approach
- Test-driven development workflow
- Feature branch strategy
- Progressive WIP commits
- Real and mock API testing

## Technical Constraints
1. API Limitations
   - OpenAI API rate limits
   - External API quotas
   - Response time requirements
   - Data freshness considerations

2. Processing Requirements
   - Real-time entity detection
   - Memory efficient context windows
   - Session state management
   - Background task handling

3. Performance Requirements
   - Sub-second entity detection
   - Real-time chat response
   - Efficient session management
   - Optimized embedding calculations
   - Background task prioritization 
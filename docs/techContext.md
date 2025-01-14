# Technical Context

## Technologies Used
- Python 3.10+
- Django 5.0+
- OpenAI API (text-embedding-ada-002)
- PostgreSQL
- pytest for testing
- numpy for numerical operations
- scikit-learn for similarity calculations

## Development Setup
- Async-first development approach
- Test-driven development workflow
- Real and mock API testing capabilities
- Background task processing

## Technical Constraints
1. **API Limitations**
   - OpenAI API rate limits
   - Embedding model constraints
   - Response time requirements

2. **Performance Requirements**
   - Real-time streaming processing
   - Memory efficient context window
   - Efficient database operations

3. **Testing Constraints**
   - API key required for real tests
   - Mock tests for development
   - Consistent test environment

4. **Database Considerations**
   - Efficient embedding storage
   - Index optimization
   - Transaction management 
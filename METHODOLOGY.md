# Development Methodology Specification


## 0. Context
- Don't create new files unless you ask
- Ensure that you've been given the right files in your context so that you can work on the right code
- Before creating new files ensure we have a commit that we can revert to if you need to


## 1. Technology Stack
- Django Framework:
  - Async views where possible
  - View choice based on use case:
    * Function-based views: Simple operations, streaming responses
    * Class-based views: Complex CRUD, form handling, reusable patterns
  - Django templates with HTMX
- Frontend Strategy:
  - HTMX for dynamic interactions
  - Minimal JavaScript (only when necessary)
  - Server-side rendering preferred
- Development Tools:
  - livereload for auto-reloading:
    ```bash
    python manage.py livereload
    ```
  - httpx for async HTTP requests

## 2. Incremental Code Development
- Each code segment must be:
  - Self-contained where possible
  - Well-commented
  - Reviewed before proceeding
- Async-first approach:
  - Use async views with `async def`
  - Leverage `httpx` for external requests
  - Handle blocking operations with `sync_to_async`

## 3. Test-Driven Development
- Write pytest tests for new functionality
- Use pytest-watch (ptw) during development:
  ```bash
  ptw -- -vv
  ```
- Test categories:
  - Unit tests for individual functions
  - Integration tests for component interaction
  - Async tests for streaming functionality
  - HTMX interaction tests
  - Mock external services (httpx requests)
- Test files mirror source structure:
  ```
  src/
    module/
      file.py
  tests/
    module/
      test_file.py
  ```

## 4. Documentation Management
- Maintain a living specifications document
- Document sections:
  - Current implementation status
  - Planned features/changes
  - Known issues/limitations
  - Dependencies
  - Configuration requirements
- Update specs after each successful iteration

## 5. Version Control Strategy
- Feature Branch Workflow:
  - Create feature branch from main/master
  - Naming convention: `feature/descriptive-name`
  - Regular WIP commits with meaningful messages
  - Format: `[WIP] component: brief description`
- Commit checkpoints after each working state

## 6. Verification Process
- Terminal Commands Checklist:
  ```bash
  # Terminal 1: Django development server
  ./manage.py runserver

  # Terminal 2: Live reload
  ./manage.py livereload

  # Terminal 3: Pytest watch
  ptw -- -vv
  ```
- Document successful command sequences
- Track error messages and resolutions
- Verify HTMX interactions in browser dev tools

## 7. Progress Tracking
- Verify each step before proceeding:
  - Tests pass
  - Code runs without errors
  - Feature works as intended
  - Documentation is updated
  - Changes are committed
- Review points:
  - Code quality and style
  - Test coverage
  - Performance metrics
  - Security considerations
  - Accessibility compliance

### Testing Strategy: Mock vs Real API Calls
- Use configurable fixtures for external services:
  ```python
  @pytest.fixture
  async def openai_client(request):
      if request.config.getoption("--use-real-api"):
          return AsyncOpenAI()
      return AsyncMock()  # Configured mock
  ```

- Test execution with pytest-watch:
  ```bash
  # Run with mocks (fast, default)
  ptw -- -vv

  # Run with real API
  ptw -- -vv --use-real-api

  # Run specific test with real API
  ptw tests/test_chat.py::test_chat_response -- -vv --use-real-api
  ```

- Test organization:
  ```
  tests/
    conftest.py          # Shared fixtures and config
    test_chat.py         # Tests using fixtures
    mocks/
      openai_responses/  # Mock response data
  ```

- Development workflow:
  1. Activate poetry shell once per session
  2. Write tests using fixtures
  3. Run ptw with mocks during development
  4. Verify with real API before committing
  5. CI pipeline uses mocks by default

- Benefits:
  * Fast feedback loop with mocks
  * Real API verification when needed
  * Consistent test code
  * Flexible local development
  * Reliable CI pipeline
  * Continuous test running with ptw

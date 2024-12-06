# Artefact Project

A Django-based chat application with AI integration and real-time streaming capabilities.

## Quick Start
1. Clone the repository
   ```bash
   git clone https://github.com/yeshuman/artefact.git
   cd artefact
   ```

2. Set up environment with Poetry
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies and create virtual environment
   poetry install

   # Activate the virtual environment
   poetry shell
   ```

3. Configure environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. Run development servers
   ```bash
   # Terminal 1: Django development server
   poetry run python manage.py runserver

   # Terminal 2: Live reload
   poetry run python manage.py livereload

   # Terminal 3: Pytest watch
   poetry run ptw -- -vv
   ```

## Development Setup
- Python 3.11+
- Poetry for dependency management
- Django 5.0+
- Node.js (for build tools)
- OpenAI API key

## Dependencies Management
- Add new dependencies: `poetry add package_name`
- Add dev dependencies: `poetry add --group dev package_name`
- Update dependencies: `poetry update`
- Export requirements: `poetry export -f requirements.txt --output requirements.txt`

## AI-Assisted Development
This project uses AI-assisted development with Cursor editor:
1. Install Cursor: https://cursor.sh
2. Copy contents of `METHODOLOGY.md` into Cursor's AI Rules
3. Follow development methodology for consistency

## Project Structure
```
artefact/
├── chats/              # Chat application
├── static/             # Static files
├── templates/          # HTML templates
├── tests/              # Test suite
└── artefact/           # Project settings
```

## Contributing
- See `METHODOLOGY.md` for development approach
- See `SPECIFICATIONS.md` for feature details

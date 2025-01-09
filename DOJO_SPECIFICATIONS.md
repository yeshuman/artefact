# Dojo System Specifications

## Overview
The Dojo system implements a dynamic learning environment where a Ronin (seeker) and Satori (guide) engage in meaningful dialogue while detecting and learning from entities that emerge in their conversations.

## Terminal Interface Design

### Layout Structure
```
╔════════════════════════════════════════════════════════════════╗
║                  ARTEFACT DOJO INITIALIZATION                   ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  📜 Dojo Configuration                                         ║
║  ══════════════════                                           ║
║  Theme: [Selected theme]                                       ║
║  Principles:                                                   ║
║  • [Principle 1]                                              ║
║  • [Principle 2]                                              ║
║                                                               ║
║  🔮 Entity Archetypes                                         ║
║  ══════════════════                                           ║
║  • [Archetype 1]: [Description]                               ║
║  • [Archetype 2]: [Description]                               ║
║                                                               ║
║  📚 Initial Reference Entities                                ║
║  ═══════════════════════                                      ║
║  • [Entity 1] (Archetype)                                     ║
║  • [Entity 2] (Archetype)                                     ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  🌟 Seeker Profile                                            ║
║  ═══════════════                                              ║
║  Ronin: [Name]                                                ║
║  Style: [Style]                                               ║
║  Interests: [Interest 1], [Interest 2]                        ║
║                                                               ║
║  🌸 Guide Profile                                             ║
║  ══════════════                                               ║
║  Satori: [Name]                                               ║
║  Style: [Style]                                               ║
║  Specialties: [Specialty 1], [Specialty 2]                    ║
╠════════════════════════════════════════════════════════════════╣
║                      MONDO DIALOGUE                            ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  [Ronin]: [Question text with highlighted entities]            ║
║  Found new entity: [Entity] (Archetype)                        ║
║                                                                ║
║  [Satori]: [Response text with highlighted entities]           ║
║  Found new entity: [Entity] (Archetype)                        ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║                    MONDO COMPLETE                              ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  🔍 Entities Discovered                                        ║
║  ══════════════════                                           ║
║  • [Entity 1]                                                 ║
║  • [Entity 2]                                                 ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### Color Scheme
- Headers: Bold Green
- Panels: Blue borders
- Ronin messages: Yellow
- Satori messages: Green
- Entities: Bold Cyan
- Entity notifications: Dim White
- Section dividers: White

## Output Configuration

### Logging Levels
```python
# Custom logger configuration
class DojoLogger:
    def __init__(self, console: Console):
        self.console = console
        self.verbose_mode = False
    
    def set_verbose(self, verbose: bool):
        self.verbose_mode = verbose
    
    def debug(self, message: str):
        """Development debugging messages."""
        if self.verbose_mode:
            self.console.print(f"[dim]DEBUG: {message}[/dim]")
    
    def info(self, message: str):
        """Standard information messages."""
        self.console.print(message)
    
    def entity(self, entity: str, archetype: str):
        """Entity detection notifications."""
        if not self.verbose_mode:
            self.console.print(f"[dim]Found new entity: {entity} ({archetype})[/dim]")
        else:
            self.console.print(f"[dim]Entity detected: {entity} ({archetype}) - confidence: {confidence}[/dim]")
    
    def system(self, message: str):
        """System state messages."""
        if self.verbose_mode:
            self.console.print(f"[blue]{message}[/blue]")
```

### Usage in Command
```python
# In run_dojo.py
class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )
    
    async def async_handle(self, *args, **options):
        console = Console()
        logger = DojoLogger(console)
        logger.set_verbose(options['verbose'])
        
        # Use logger instead of print/logging
        logger.system("Initializing Dojo...")
        logger.info("Creating new session...")
        logger.debug("Loading configuration...")
```

## Core Components

### 1. Dojo Initialization
- Self-selects theme and principles during initialization
- Creates entity archetypes based on the chosen theme
- Establishes initial reference entities
- Initializes Ronin and Satori with appropriate personalities

### 2. Entity Detection System
- Real-time entity detection during conversations
- Two-phase matching approach:
  1. Match against archetypes first
  2. Match against existing reference entities
- Confidence threshold for reference entity creation (>= 0.95)
- Support for multi-word entity detection
- Entity position tracking in messages

### 3. Reference Entity Management
- Global reference entities (mondo=None)
- Mondo-specific reference entities
- High-confidence matches become new reference entities
- Progressive learning through conversation

### 4. Conversation Flow
- Ronin initiates with questions
- Satori responds with guidance
- Entity detection runs on all messages
- Ronin determines when to end conversation
- Streaming output with real-time entity highlighting

## Technical Implementation

### Entity Detection
```python
class StreamingEntityDetector:
    async def process_chunk(self, text: str, start_pos: int) -> List[Entity]:
        # Process text chunks for entities
        # Return entities with position information

    async def _find_entities(self, text: str) -> List[Entity]:
        # Two-phase matching:
        # 1. Match against archetypes
        # 2. Match against reference entities
```

## Testing Strategy

### Core Test Cases
1. Entity Detection
   - Word boundary detection
   - Context window handling
   - Multi-word entity detection
   - Position tracking accuracy

2. Reference Entity Storage
   - Confidence threshold validation
   - High-confidence storage
   - Low-confidence rejection

3. Conversation Flow
   - Message creation
   - Response generation
   - Entity highlighting
   - Conversation termination

### Test Implementation
```python
async def test_streaming_entity_detector_word_boundary():
    # Verify correct word boundary handling

async def test_streaming_entity_detector_context_window():
    # Verify context window processing

async def test_streaming_entity_detector_markup():
    # Verify entity markup generation

async def test_high_confidence_reference_creation():
    # Verify reference entity storage based on confidence

async def test_multi_word_entity_detection():
    # Verify detection of multi-word entities
```

## Usage

### Running the Dojo
```bash
python manage.py run_dojo
```

### Expected Output
1. Initialization Phase
   - Dojo theme and principles
   - Created archetypes
   - Initial reference entities
   - Character initialization

2. Conversation Phase
   - Real-time message display
   - Entity highlighting
   - Entity detection notifications

3. Summary Phase
   - List of discovered entities
   - Conversation statistics

## Future Enhancements
1. Enhanced entity type support
2. Improved confidence calculation
3. Entity relationship mapping
4. Historical context awareness
5. Interactive entity exploration 

### Logging Configuration

#### Dual Logging Approach
1. Standard Debug Logger
   - Maintains existing `logging` module configuration
   - Used for development and debugging
   - Controlled by log level settings
   - Outputs to configured handlers (file, console, etc.)

2. Presentation Logger (DojoPresenter)
```python
# Custom presenter for terminal UI
class DojoPresenter:
    def __init__(self, console: Console):
        self.console = console
        self.verbose_mode = False
    
    def set_verbose(self, verbose: bool):
        """Control detail level of terminal output."""
        self.verbose_mode = verbose
    
    def show_debug(self, message: str):
        """Optional development details."""
        if self.verbose_mode:
            self.console.print(f"[dim]DEBUG: {message}[/dim]")
    
    def show_message(self, message: str, style: str = None):
        """Display formatted message in terminal."""
        if style:
            self.console.print(f"[{style}]{message}[/{style}]")
        else:
            self.console.print(message)
    
    def show_entity(self, entity: str, archetype: str, confidence: float = None):
        """Display entity detection in terminal."""
        if not self.verbose_mode:
            self.console.print(f"[dim]Found new entity: {entity} ({archetype})[/dim]")
        else:
            self.console.print(f"[dim]Entity detected: {entity} ({archetype}) - confidence: {confidence:.4f}[/dim]")
    
    def show_panel(self, content: str, title: str = None, style: str = "blue"):
        """Display boxed content."""
        self.console.print(Panel(content, title=title, border_style=style))

### Usage in Command
```python
# In run_dojo.py
class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose terminal output'
        )
    
    async def async_handle(self, *args, **options):
        # Standard debug logger
        logger = logging.getLogger(__name__)
        logger.debug("Starting Dojo initialization")
        
        # Terminal presentation
        console = Console()
        presenter = DojoPresenter(console)
        presenter.set_verbose(options['verbose'])
        
        # Example usage
        logger.debug("Loading Dojo configuration")  # Debug log
        presenter.show_message("Creating new Dojo...", style="bold blue")  # Terminal UI
        
        # Initialize Dojo
        dojo_service = DojoService(llm_client=AsyncOpenAI())
        await dojo_service.initialize()
        logger.debug("Dojo initialized with theme: %s", dojo_service.model.theme)
        
        # Show configuration in terminal
        presenter.show_panel(
            f"[bold]Theme:[/bold] {dojo_service.model.theme}\n\n" +
            "[bold]Principles:[/bold]\n" +
            "\n".join(f"• {p}" for p in dojo_service.model.principles),
            title="📜 Dojo Configuration"
        )
```

This approach:
- Keeps existing debug logging intact
- Adds pretty terminal output for presentations
- Separates concerns between debugging and presentation
- Allows verbose mode for additional terminal details
- Maintains consistent styling through the presenter 
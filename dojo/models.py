from django.db import models
from django.contrib.postgres.fields import ArrayField


class Dojo(models.Model):
    """A space for learning and discovery through dialogue.
    
    The Dojo (道場) establishes the cultural and philosophical context
    for interactions between Ronin and Satori. It shapes their personalities,
    communication styles, and the nature of their dialogue.
    """
    
    # Cultural and philosophical context
    theme = models.TextField(
        help_text="The overarching theme or atmosphere of this dojo"
    )
    principles = ArrayField(
        models.TextField(),
        help_text="Core principles that guide interactions",
        default=list
    )
    
    # System messages for personality shaping
    ronin_system_message = models.TextField(
        help_text="System message that shapes the Ronin's personality and approach"
    )
    satori_system_message = models.TextField(
        help_text="System message that shapes the Satori's personality and approach"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Dojo {self.id} - {self.theme}"

    @property
    async def atheme(self):
        """Async access to theme field."""
        return self.theme

    @property
    async def aprinciples(self):
        """Async access to principles field."""
        return self.principles

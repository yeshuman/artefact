from django.db import models
from django.contrib.postgres.fields import ArrayField


class Ronin(models.Model):
    """A seeker in search of wisdom and understanding"""
    name = models.CharField(max_length=100)
    interests = ArrayField(
        models.TextField(),
        help_text="List of interests",
        null=True
    )
    style = models.TextField(
        help_text="Personal approach and characteristics",
        null=True
    )
    system_prompt = models.TextField(
        help_text="Dynamic system prompt based on dojo theme",
        null=True
    )
    meditation_prompt = models.TextField(
        help_text="Dynamic meditation prompt based on dojo theme",
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Ronin"
        verbose_name_plural = "Ronin"

    def __str__(self):
        return f"{self.name} - {self.style} seeker"

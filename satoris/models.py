from django.db import models


class Satori(models.Model):
    """A guide who illuminates the path to understanding"""
    
    name = models.CharField(max_length=100)
    specialties = models.JSONField(
        default=list,
        help_text="Areas of expertise"
    )
    teaching_style = models.TextField(
        help_text="Natural approach to guiding others",
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
    model_name = models.CharField(
        max_length=100,
        default="gpt-4-1106-preview",
        help_text="The OpenAI model to use for this Satori"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Satori"
        verbose_name_plural = "Satori"

    def __str__(self):
        return f"{self.name} - Enlightened Guide"

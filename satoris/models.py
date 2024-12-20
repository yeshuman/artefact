from django.db import models


class Satori(models.Model):
    """An enlightened guide who illuminates the world's artifacts with wisdom"""
    
    name = models.CharField(max_length=100)
    specialties = models.JSONField(
        default=list,
        help_text="Areas of expertise (e.g., historical sites, cultural experiences)"
    )
    teaching_style = models.TextField(
        help_text="Natural approach to guiding others",
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Satori"
        verbose_name_plural = "Satori"

    def __str__(self):
        return f"{self.name} - Enlightened Guide"

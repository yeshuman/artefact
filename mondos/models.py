from django.db import models


class Mondo(models.Model):
    """A dialogue between Ronin and Satori within a quest.
    
    In the Zen tradition, a Mondō (問答) is a spiritual dialogue between
    seeker and master. Here, it represents the exchange between Ronin
    (the seeker) and Satori (the guide) in their journey of discovery.
    """
    quest = models.ForeignKey('quests.Quest', on_delete=models.CASCADE, related_name='mondos')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Mondo in {self.quest.title}"


class Message(models.Model):
    """A message within a Mondo dialogue."""
    mondo = models.ForeignKey(Mondo, related_name='messages', on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.content[:50]}..."


class RoninMessage(Message):
    """A question or reflection from the Ronin."""
    author = models.ForeignKey('ronins.Ronin', on_delete=models.CASCADE, related_name='messages')


class SatoriMessage(Message):
    """An illuminating response from the Satori."""
    author = models.ForeignKey('satoris.Satori', on_delete=models.CASCADE, related_name='messages')

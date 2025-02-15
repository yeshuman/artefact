from django.db import models


class Quest(models.Model):
    """A journey of discovery between a Ronin and Satori"""
    dojo = models.ForeignKey(
        'dojo.Dojo',
        on_delete=models.CASCADE,
        related_name='quests'
    )
    ronin = models.ForeignKey(
        'ronins.Ronin',
        on_delete=models.CASCADE,
        related_name='quests'
    )
    satori = models.ForeignKey(
        'satoris.Satori',
        on_delete=models.CASCADE,
        related_name='guidances'
    )
    title = models.CharField(
        max_length=200,
        help_text="Main focus of the quest (e.g., 'Exploring Ancient Kyoto')"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} - {self.ronin.name} guided by {self.satori.name}"

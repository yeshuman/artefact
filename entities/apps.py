from django.apps import AppConfig


class EntitiesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'entities'
    
    def ready(self):
        """Register signals when the app is ready."""
        import entities.signals  # noqa

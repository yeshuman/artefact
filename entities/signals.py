import logging
from django.db.models.signals import post_save
from django.dispatch import receiver
from asgiref.sync import async_to_sync
from mondos.models import Message
from .detection import detect_entities

logger = logging.getLogger(__name__)

@receiver(post_save, sender=Message)
def detect_message_entities(sender, instance, created, **kwargs):
    """
    Signal handler to detect entities in newly created messages.
    """
    if created:
        logger.info(f"Detecting entities for message {instance.id}")
        try:
            # Run entity detection asynchronously
            entities = async_to_sync(detect_entities)(instance)
            logger.info(f"Detected {len(entities)} entities in message {instance.id}")
        except Exception as e:
            logger.error(f"Error detecting entities in message {instance.id}: {e}") 
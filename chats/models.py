import uuid
from django.db import models
from polymorphic.models import PolymorphicModel


class Chat(models.Model):

    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Chat {self.uuid}"


class Message(PolymorphicModel):
    # chat = models.ForeignKey(Chat, on_delete=models.CASCADE)
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message {self.id} from {self.chat.uuid}"
    
    class Meta:
        ordering = ['-created_at']


class HumanMessage(Message):
    # user = models.ForeignKey(User, on_delete=models.CASCADE)
    pass



class BotMessage(Message):
    pass

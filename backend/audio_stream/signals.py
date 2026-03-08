from django.db.models.signals import post_save
from django.dispatch import receiver
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from .models import Face

@receiver(post_save, sender=Face)
def on_face_updated(sender, instance, **kwargs):
    """
    Broadcasts face updates to all connected clients.
    """
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        'face_updates',
        {
            'type': 'face.update',
            'face_id': instance.id,
            'label': instance.label,
            'name': instance.name or 'Identifying...',
            'relationship': instance.relationship or 'Known Person',
            'metadata': instance.metadata or []
        }
    )

from django.db import models
import json

class Face(models.Model):
    label = models.CharField(max_length=100)
    descriptor = models.TextField() # Stored as JSON string (Float32Array)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_descriptor(self, data):
        self.descriptor = json.dumps(data)

    def get_descriptor(self):
        return json.loads(self.descriptor)

    def __str__(self):
        return self.label

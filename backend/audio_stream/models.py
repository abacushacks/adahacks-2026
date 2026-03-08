from django.db import models
from typing import List, Any
import json

class Face(models.Model):
    """
    Represents a face descriptor and its associated label for recognition.
    """
    label = models.CharField(max_length=100)
    descriptor = models.TextField()  # Stored as JSON string (Float32Array)
    created_at = models.DateTimeField(auto_now_add=True)

    def set_descriptor(self, data: List[float]) -> None:
        """
        Serializes the face descriptor list into a JSON string.
        """
        self.descriptor = json.dumps(data)

    def get_descriptor(self) -> List[float]:
        """
        Deserializes the face descriptor JSON string back into a list of floats.
        """
        return json.loads(self.descriptor)

    def __str__(self) -> str:
        return self.label

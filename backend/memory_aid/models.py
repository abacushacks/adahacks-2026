from django.db import models


class Person(models.Model):
    face_embedding_data = models.JSONField()
    name = models.CharField(max_length=255)

    def __str__(self) -> str:
        return self.name


class Memory(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name="memories")
    fact_type = models.CharField(max_length=64)
    fact_value = models.TextField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("person", "fact_type", "fact_value"),
                name="unique_person_fact",
            )
        ]

    def __str__(self) -> str:
        return f"{self.person.name}: {self.fact_type} -> {self.fact_value}"

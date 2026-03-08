from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Person",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("face_embedding_data", models.JSONField()),
                ("name", models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name="Memory",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("fact_type", models.CharField(max_length=64)),
                ("fact_value", models.TextField()),
                (
                    "person",
                    models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="memories", to="memory_aid.person"),
                ),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(
                        fields=("person", "fact_type", "fact_value"),
                        name="unique_person_fact",
                    )
                ]
            },
        ),
    ]

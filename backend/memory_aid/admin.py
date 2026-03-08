from django.contrib import admin

from memory_aid.models import Memory, Person


class MemoryInline(admin.TabularInline):
    model = Memory
    extra = 0


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ("id", "name")
    search_fields = ("name",)
    inlines = [MemoryInline]


@admin.register(Memory)
class MemoryAdmin(admin.ModelAdmin):
    list_display = ("id", "person", "fact_type", "fact_value")
    list_filter = ("fact_type",)
    search_fields = ("person__name", "fact_value")

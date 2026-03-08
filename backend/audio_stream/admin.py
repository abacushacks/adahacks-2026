from django.contrib import admin
from .models import Face

@admin.register(Face)
class FaceAdmin(admin.ModelAdmin):
    list_display = ('label', 'created_at')
    search_fields = ('label',)

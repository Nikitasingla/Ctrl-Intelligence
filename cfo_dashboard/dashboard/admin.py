from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import UploadedDocument, FinancialData, ProcessingJob

@admin.register(UploadedDocument)
class UploadedDocumentAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'file_type', 'user', 'uploaded_at', 'processed']
    list_filter = ['file_type', 'processed', 'uploaded_at']
    search_fields = ['original_filename', 'user__username']
    readonly_fields = ['uploaded_at']

@admin.register(FinancialData)
class FinancialDataAdmin(admin.ModelAdmin):
    list_display = ['document', 'competitor_ticker', 'created_at']
    list_filter = ['created_at', 'competitor_ticker']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(ProcessingJob)
class ProcessingJobAdmin(admin.ModelAdmin):
    list_display = ['document', 'status', 'progress', 'started_at', 'completed_at']
    list_filter = ['status', 'started_at']
    readonly_fields = ['started_at', 'completed_at']
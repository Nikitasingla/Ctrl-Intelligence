from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
import json

class UploadedDocument(models.Model):
    DOCUMENT_TYPES = [
        ('pdf', 'PDF Document'),
        ('csv', 'CSV File'),
        ('excel', 'Excel File'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    file = models.FileField(upload_to='documents/')
    file_type = models.CharField(max_length=10, choices=DOCUMENT_TYPES)
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    processing_error = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.original_filename} - {self.uploaded_at.strftime('%Y-%m-%d')}"

class FinancialData(models.Model):
    document = models.OneToOneField(UploadedDocument, on_delete=models.CASCADE)
    extracted_data = models.JSONField()
    competitor_ticker = models.CharField(max_length=20, blank=True, null=True)
    competitor_data = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def get_historical_performance(self):
        return self.extracted_data.get('historical_performance', [])
    
    def get_revenue_breakdown(self):
        return self.extracted_data.get('revenue_breakdown_crore', [])
    
    def get_risks(self):
        return self.extracted_data.get('future_risks', [])
    
    def get_opportunities(self):
        return self.extracted_data.get('future_opportunities', [])

class ProcessingJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    document = models.OneToOneField(UploadedDocument, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, null=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Message(models.Model):
    ROLE_CHOICES = [('user','user'),('assistant','assistant'),('system','system')]
    conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    metadata = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
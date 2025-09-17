from django.urls import path
from . import views
from .views import ChatAPIView, chat_page
app_name = 'dashboard'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_view, name='upload'),
    path('processing/<int:job_id>/', views.processing_status, name='processing_status'),
    path('dashboard/<int:document_id>/', views.dashboard_view, name='dashboard_view'),
    path('sample/', views.sample_dashboard, name='sample_dashboard'),
    
    # API endpoints
    path('api/risk-explanation/', views.get_risk_explanation, name='risk_explanation'),
    path('api/opportunity-explanation/', views.get_opportunity_explanation, name='opportunity_explanation'),
    path('api/what-if-analysis/', views.what_if_analysis, name='what_if_analysis'),
    path('api/competitor-insights/<int:document_id>/', views.get_competitor_insights, name='competitor_insights'),


    path('api/runway-what-if/', views.runway_what_if, name='runway_what_if'),
    path("api/chat/", ChatAPIView.as_view(), name="chat"),
    path("webchat/", chat_page, name="webchat")
]
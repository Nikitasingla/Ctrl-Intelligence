from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import UploadedDocument, FinancialData, ProcessingJob
from .forms import DocumentUploadForm
from .utils import (
    DocumentProcessor, CompetitorAnalyzer, RiskOpportunityAnalyzer,
    advanced_forecast, enhanced_competitor_analysis, FinancialAnalyzer,
    generate_executive_summary, calculate_what_if_runway # ADD a new import
)
import json
import os
import threading
import plotly.express as px
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def home(request):
    return render(request, 'dashboard/home.html')


def upload_view(request):
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)
            document.user = request.user if request.user.is_authenticated else None
            document.original_filename = document.file.name
            file_extension = os.path.splitext(document.file.name)[1].lower()
            if file_extension == '.pdf':
                document.file_type = 'pdf'
            elif file_extension == '.csv':
                document.file_type = 'csv'
            else:
                document.file_type = 'excel'
            document.save()
            job = ProcessingJob.objects.create(document=document)
            competitor_ticker = form.cleaned_data.get('competitor_ticker')
            thread = threading.Thread(
                target=process_document_background,
                args=(document.id, competitor_ticker)
            )
            thread.daemon = True
            thread.start()
            messages.success(request, 'File uploaded successfully! Processing started...')
            return redirect('dashboard:processing_status', job_id=job.id)
    else:
        form = DocumentUploadForm()
    return render(request, 'dashboard/upload.html', {'form': form})


def process_document_background(document_id, competitor_ticker=None):
    job = None
    document = None
    try:
        document = UploadedDocument.objects.get(id=document_id)
        job = ProcessingJob.objects.get(document=document)
        job.status = 'processing'
        job.save()

        processor = DocumentProcessor()
        if document.file_type == 'pdf':
            processed_data = processor.process_pdf(document.file.path)
            extracted_data = processor.extract_financial_data(processed_data)
        else:
            extracted_data = processor.process_structured_file(document.file.path, document.file_type)

        FinancialData.objects.create(
            document=document,
            extracted_data=extracted_data,
            competitor_ticker=competitor_ticker,
        )

        document.processed = True
        document.save()
        job.status = 'completed'
        job.progress = 100
        job.save()
    except Exception as e:
        logger.error(f"Processing error: {e}")
        if job:
            job.status = 'failed'
            job.error_message = str(e)
            job.save()
        if document:
            document.processed = True
            document.processing_error = str(e)
            document.save()


def processing_status(request, job_id):
    job = get_object_or_404(ProcessingJob, id=job_id)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'status': job.status,
            'progress': job.progress,
            'error': job.error_message,
            'completed': job.status in ['completed', 'failed']
        })
    if job.status == 'completed' and not job.document.processing_error:
        return redirect('dashboard:dashboard_view', document_id=job.document.id)
    return render(request, 'dashboard/processing.html', {'job': job})


def dashboard_view(request, document_id):
    document = get_object_or_404(UploadedDocument, id=document_id)
    if not document.processed:
        messages.error(request, 'Document is still being processed.')
        return redirect('dashboard:upload')
    if document.processing_error:
        messages.error(request, f'Processing failed: {document.processing_error}')
        return redirect('dashboard:upload')

    try:
        financial_data = FinancialData.objects.get(document=document)
    except FinancialData.DoesNotExist:
        messages.error(request, 'No financial data found for this document.')
        return redirect('dashboard:upload')

    historical_data = financial_data.get_historical_performance()
    can_forecast = historical_data and len(historical_data) >= 2

    analyzer = FinancialAnalyzer(financial_data.extracted_data)
    analysis_results = analyzer.run_analysis()

    # Generate AI Executive Summary
    executive_summary = generate_executive_summary(
        financial_data.extracted_data,
        analysis_results,
        settings.GEMINI_API_KEY
    )

    revenue_breakdown = financial_data.get_revenue_breakdown()
    risks = financial_data.get_risks()
    opportunities = financial_data.get_opportunities()
    competitor_data = None
    if financial_data.competitor_ticker and historical_data:
        try:
            competitor_data = enhanced_competitor_analysis(financial_data.competitor_ticker, historical_data)
        except Exception as e:
            logger.error(f"Error loading competitor data: {e}")

    charts = create_dashboard_charts(historical_data, revenue_breakdown, competitor_data)
    forecast_data = []
    if can_forecast:
        try:
            forecast_data = advanced_forecast(historical_data)
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")

    # Prepare data sources for tooltips
    data_sources = {item['metric']: item['source'] for item in financial_data.extracted_data.get('data_sources', [])}

    context = {
        'document': document,
        'financial_data': financial_data,
        'historical_data': historical_data,
        'revenue_breakdown': revenue_breakdown,
        'risks': risks,
        'opportunities': opportunities,
        'charts': charts,
        'has_competitor_data': bool(financial_data.competitor_ticker),
        'competitor_data': competitor_data,
        'historical_data_json': json.dumps(historical_data),
        'forecast_data_json': json.dumps(forecast_data),
        'can_forecast': can_forecast,
        'analysis_results': analysis_results,
        'executive_summary': executive_summary,
        'data_sources': data_sources,
    }
    return render(request, 'dashboard/enhanced_dashboard.html', context)


def sample_dashboard(request):
    sample_data = {
        "historical_performance": [
            {"year": 2023, "revenue_crore": 8298, "profit_crore": -346},
            {"year": 2024, "revenue_crore": 12114, "profit_crore": 175}
        ],
        "revenue_breakdown_crore": [{"segment": "Food Delivery", "revenue": 8500}, {"segment": "Hyperpure", "revenue": 1614}],
        "future_risks": ["Intense competition", "Economic downturns"],
        "future_opportunities": ["Market expansion", "Quick commerce growth"],
        "balance_sheet": [
            {"year": 2023, "cash_and_equivalents_crore": 11000, "current_assets_crore": 13000, "total_assets_crore": 25000, "current_liabilities_crore": 4000, "total_liabilities_crore": 5000},
            {"year": 2024, "cash_and_equivalents_crore": 12000, "current_assets_crore": 15000, "total_assets_crore": 28000, "current_liabilities_crore": 13000, "total_liabilities_crore": 14000}
        ],
        "cash_flow": [],
        "data_sources": [
            { "metric": "Revenue 2024", "source": "Sample Data, Annual Report FY24" },
            { "metric": "Profit 2024", "source": "Sample Data, P&L Statement FY24" }
        ]
    }

    analyzer = FinancialAnalyzer(sample_data)
    analysis_results = analyzer.run_analysis()

    charts = create_dashboard_charts(sample_data["historical_performance"], sample_data["revenue_breakdown_crore"], None)
    forecast_data = advanced_forecast(sample_data["historical_performance"])

    data_sources = {item['metric']: item['source'] for item in sample_data.get('data_sources', [])}

    context = {
        'sample_mode': True,
        'document': {'id': 0, 'original_filename': 'Sample Data (Zomato)'},
        'historical_data': sample_data["historical_performance"],
        'revenue_breakdown': sample_data["revenue_breakdown_crore"],
        'risks': sample_data["future_risks"],
        'opportunities': sample_data["future_opportunities"],
        'charts': charts,
        'has_competitor_data': False,
        'financial_data': {'competitor_ticker': None},
        'historical_data_json': json.dumps(sample_data["historical_performance"]),
        'forecast_data_json': json.dumps(forecast_data),
        'can_forecast': True,
        'analysis_results': analysis_results,
        'data_sources': data_sources,
        'executive_summary': "This is a sample AI-generated summary. The company shows strong revenue growth but needs to manage its high burn rate to ensure long-term stability."
    }
    return render(request, 'dashboard/enhanced_dashboard.html', context)


# def create_dashboard_charts(historical_data, revenue_breakdown, competitor_data):
#     charts = {}
#     if not historical_data:
#         return charts
    
#     try:
#         df = pd.DataFrame(historical_data)
#         fig = px.bar(
#             df, x='year', y=['revenue_crore', 'profit_crore'],
#             barmode='group', title='Historical Revenue vs. Profit (₹ Crore)',
#             labels={'value': 'Amount (₹ Crore)', 'year': 'Year'}
#         )
#         charts['historical'] = fig.to_html(include_plotlyjs=False, div_id="historical-chart")

#         if revenue_breakdown:
#             df_breakdown = pd.DataFrame(revenue_breakdown)
#             fig_pie = px.pie(
#                 df_breakdown, names='segment', values='revenue',
#                 title='Revenue Breakdown by Segment', hole=0.4
#             )
#             charts['breakdown'] = fig_pie.to_html(include_plotlyjs=False, div_id="breakdown-chart")

#     except Exception as e:
#         logger.error(f"Error creating basic charts: {e}")

#     return charts
def create_dashboard_charts(historical_data, revenue_breakdown, competitor_data):
    charts = {}
    if not historical_data:
        return charts

    try:
        df = pd.DataFrame(historical_data)

        # --- FIX: Standardize all column names to lowercase ---
        df.columns = df.columns.str.lower()

        # For debugging: print the available columns to the terminal
        print("--- Columns available for charting:", df.columns.tolist())

        # Define the columns we absolutely need for the chart
        required_cols = ['year', 'revenue_crore', 'profit_crore']

        # Check if all required columns are present
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Chart creation skipped: Missing one or more required columns: {required_cols}")
            # Provide a helpful message to display on the dashboard
            charts['historical'] = "<div class='alert alert-warning'>Chart could not be generated. The uploaded file might be missing the 'Year', 'Revenue Crore', or 'Profit Crore' columns.</div>"
            return charts

        # If we have the columns, create the chart
        fig = px.bar(
            df, x='year', y=['revenue_crore', 'profit_crore'],
            barmode='group', title='Historical Revenue vs. Profit (₹ Crore)',
            labels={'value': 'Amount (₹ Crore)', 'year': 'Year'}
        )
        charts['historical'] = fig.to_html(include_plotlyjs=False, div_id="historical-chart")

        if revenue_breakdown:
            try:
                df_breakdown = pd.DataFrame(revenue_breakdown)
                df_breakdown.columns = df_breakdown.columns.str.lower()
                if 'segment' in df_breakdown.columns and 'revenue' in df_breakdown.columns:
                    fig_pie = px.pie(
                        df_breakdown, names='segment', values='revenue',
                        title='Revenue Breakdown by Segment', hole=0.4
                    )
                    charts['breakdown'] = fig_pie.to_html(include_plotlyjs=False, div_id="breakdown-chart")
            except Exception as e:
                logger.error(f"Error creating breakdown chart: {e}")

    except Exception as e:
        logger.error(f"A critical error occurred in create_dashboard_charts: {e}")
        charts['historical'] = "<div class='alert alert-danger'>An unexpected error occurred while generating the chart. Please check the server logs.</div>"

    return charts

@csrf_exempt
def what_if_analysis(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'})
    try:
        data = json.loads(request.body)
        document_id = data.get('document_id')
        revenue_growth = float(data.get('revenue_growth', 5))
        expense_growth = float(data.get('expense_growth', 3))
        
        document = UploadedDocument.objects.get(id=document_id)
        financial_data = FinancialData.objects.get(document=document)
        historical_data = financial_data.get_historical_performance()
        if not historical_data or len(historical_data) < 2:
            return JsonResponse({'error': 'Insufficient historical data for analysis'})

        what_if_data = advanced_forecast(
            historical_data,
            years_to_forecast=3,
            revenue_growth_rate=revenue_growth,
            expense_growth_rate=expense_growth
        )
        return JsonResponse({'success': True, 'forecast_data': what_if_data})
    except Exception as e:
        logger.error(f"What-if analysis error: {e}")
        return JsonResponse({'error': str(e)})


@csrf_exempt
def runway_what_if(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'})
    try:
        data = json.loads(request.body)
        document_id = data.get('document_id')
        
        document = get_object_or_404(UploadedDocument, id=document_id)
        financial_data = get_object_or_404(FinancialData, document=document)
        analyzer = FinancialAnalyzer(financial_data.extracted_data)
        analysis_results = analyzer.run_analysis()

        current_cash = financial_data.extracted_data.get('balance_sheet', [])[-1].get('cash_and_equivalents_crore')
        current_burn = analysis_results.get('metrics', {}).get('net_burn_monthly_crore')
        
        hiring_cost = float(data.get('hiring_cost', 0)) / 100
        new_mrr = float(data.get('new_mrr', 0)) / 100
        expense_reduction = float(data.get('expense_reduction', 0)) / 100
        
        new_runway = calculate_what_if_runway(
            current_cash, current_burn, hiring_cost, new_mrr, expense_reduction
        )
        
        return JsonResponse({'success': True, 'new_runway_months': new_runway})
    except Exception as e:
        logger.error(f"Runway what-if error: {e}")
        return JsonResponse({'error': str(e)})


def get_competitor_insights(request, document_id):
    try:
        document = get_object_or_404(UploadedDocument, id=document_id)
        financial_data = get_object_or_404(FinancialData, document=document)
        if not financial_data.competitor_ticker:
            return JsonResponse({'error': 'No competitor ticker specified'})

        company_data = financial_data.get_historical_performance()
        if not company_data:
            return JsonResponse({'error': 'No historical data available for comparison'})

        competitor_data = enhanced_competitor_analysis(
            financial_data.competitor_ticker,
            company_data
        )
        if not competitor_data:
            return JsonResponse({'error': 'Could not fetch competitor data'})
            
        return JsonResponse({
            'competitor_data': competitor_data,
            'company_data': company_data
        })
    except Exception as e:
        logger.error(f"Competitor insights error: {e}")
        return JsonResponse({'error': str(e)})


def get_risk_explanation(request):
    if request.method == 'POST':
        risk_topic = request.POST.get('risk_topic')
        if not risk_topic:
            return JsonResponse({'error': 'Risk topic not provided'})
        try:
            analyzer = RiskOpportunityAnalyzer()
            explanation = analyzer.get_risk_explanation(risk_topic)
            return JsonResponse({'explanation': explanation})
        except Exception as e:
            logger.error(f"Risk explanation error: {e}")
            return JsonResponse({'error': 'Failed to get explanation'})
    return JsonResponse({'error': 'Invalid request'})


def get_opportunity_explanation(request):
    if request.method == 'POST':
        opportunity_topic = request.POST.get('opportunity_topic')
        if not opportunity_topic:
            return JsonResponse({'error': 'Opportunity topic not provided'})
        try:
            analyzer = RiskOpportunityAnalyzer()
            explanation = analyzer.get_opportunity_explanation(opportunity_topic)
            return JsonResponse({'explanation': explanation})
        except Exception as e:
            logger.error(f"Opportunity explanation error: {e}")
            return JsonResponse({'error': 'Failed to get explanation'})
    return JsonResponse({'error': 'Invalid request'})




from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
from .models import Conversation, Message
from .serializers import ConversationSerializer
from .utils import generate_gemini_response, get_stock_price

SYSTEM_PROMPT = """
You are FinBot, a cautious financial assistant.
- Explain simply.
- Ask clarifying questions.
- Remind users you are not a licensed advisor.
"""

def chat_page(request):
    """
    Render the chat template.
    """
    return render(request, "dashboard/chat.html")

from .utils import generate_gemini_response, get_stock_price


# dashboard/views.py

class ChatAPIView(APIView):
    def post(self, request):
        print("--- ChatAPIView: Received POST request ---") # ADD THIS
        user = request.user if request.user.is_authenticated else None
        conv_id = request.data.get("conversation_id")
        user_message = request.data.get("message", "").strip()
        print(f"--- Message received: '{user_message}' ---") # ADD THIS

        if not user_message:
            return Response({"answer":"No message sent."})

        conversation, _ = Conversation.objects.get_or_create(
            id=conv_id, defaults={"user": user}
        )
        Message.objects.create(conversation=conversation, role="user", content=user_message)

        if user_message.lower().startswith("price "):
            ticker = user_message.split(" ")[1].upper()
            reply = get_stock_price(ticker)
        else:
            print("--- Calling Gemini API... ---") # ADD THIS
            reply = generate_gemini_response(user_message)
            print(f"--- Gemini response received: '{reply}' ---") # ADD THIS

        Message.objects.create(conversation=conversation, role="assistant", content=reply)
        return Response({"answer": reply, "conversation_id": conversation.id})

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .utils import generate_gemini_response

@csrf_exempt
def chat_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("message", "")
        if not user_message:
            return JsonResponse({"error": "No message provided"}, status=400)

        reply = generate_gemini_response(user_message)
        return JsonResponse({"answer": reply})

    return JsonResponse({"error": "Method not allowed"}, status=405)
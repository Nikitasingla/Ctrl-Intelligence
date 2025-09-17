import pandas as pd
import pdfplumber
import json
import requests
import numpy as np
import yfinance as yf
import time
import re
from django.conf import settings
from django.core.cache import cache
import logging

# This is the required logger definition
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, api_key=None):
        self.api_key = api_key or getattr(settings, 'GEMINI_API_KEY', None)

    def process_pdf(self, file_path, pages_limit=0):
        try:
            full_text = ""
            with pdfplumber.open(file_path) as pdf:
                pages_to_scan = min(pages_limit, len(pdf.pages)) if pages_limit > 0 else len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    if i >= pages_to_scan: break
                    full_text += f"\n\n--- Page {i+1} ---\n\n"
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text
            text = re.sub(r'\s+', ' ', full_text).strip()
            chunks = [text[i:i + 2000] for i in range(0, len(text), 1800)]
            return {"source_type": "PDF", "content_chunks": chunks}
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise

    def process_structured_file(self, file_path, file_type):
        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
            else:
                df = pd.read_excel(file_path)
            
            df.columns = df.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]', "", regex=True)

            is_transactional = 'transactiondate' in df.columns or 'sales' in df.columns
            is_yearly_summary = 'year' in df.columns and ('revenuecrore' in df.columns or 'cashandequivalentscrore' in df.columns)

            if is_yearly_summary:
                logger.info("Processing as Yearly Summary CSV")
                df.rename(columns={
                    'revenuecrore': 'revenue_crore',
                    'profitcrore': 'profit_crore',
                    'cashandequivalentscrore': 'cash_and_equivalents_crore',
                    'currentassetscrore': 'current_assets_crore',
                    'totalassetscrore': 'total_assets_crore',
                    'currentliabilitiescrore': 'current_liabilities_crore',
                    'totalliabilitiescrore': 'total_liabilities_crore',
                }, inplace=True, errors='ignore')

                required_hist = ['year', 'revenue_crore', 'profit_crore']
                required_bs = ['cash_and_equivalents_crore', 'current_assets_crore', 'total_assets_crore', 'current_liabilities_crore', 'total_liabilities_crore']
                for col in required_hist + required_bs:
                    if col not in df.columns:
                        df[col] = 0

                historical_performance = df[required_hist].to_dict('records')
                balance_sheet_data = df[['year'] + required_bs].to_dict('records')
                
                return {
                    "historical_performance": historical_performance,
                    "balance_sheet": balance_sheet_data,
                    "revenue_breakdown_crore": [], "cash_flow": [],
                    "future_risks": ["Data from yearly summary."], "future_opportunities": ["Analyze trends for opportunities."],
                }
            
            elif is_transactional:
                logger.info("Processing as Transactional CSV")
                synonyms = {
                    "date": ["transactiondate", "date"], "revenue": ["sales", "revenue"],
                    "expense": ["expenses", "expense"], "category": ["productcategory", "segment"]
                }
                def find_col(candidates):
                    for c in candidates:
                        if c in df.columns: return c
                    return None

                date_col, revenue_col, expense_col, category_col = find_col(synonyms["date"]), find_col(synonyms["revenue"]), find_col(synonyms["expense"]), find_col(synonyms["category"])
                
                if not all([date_col, revenue_col, expense_col]):
                    raise ValueError("Transactional CSV must contain date, sales, and expenses columns.")

                for col in [revenue_col, expense_col]:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                df['year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
                df.dropna(subset=['year'], inplace=True)
                df['year'] = df['year'].astype(int)
                df['calculated_profit'] = df[revenue_col] - df[expense_col]
                
                yearly_data = df.groupby('year').agg(
                    revenue_crore=(revenue_col, 'sum'),
                    profit_crore=('calculated_profit', 'sum')
                ).reset_index()

                yearly_data['revenue_crore'] /= 1_00_00_000
                yearly_data['profit_crore'] /= 1_00_00_000
                historical_performance = yearly_data.to_dict('records')
                
                revenue_breakdown = []
                if category_col:
                    category_revenue = df.groupby(category_col)[revenue_col].sum().reset_index()
                    category_revenue.rename(columns={category_col: 'segment', revenue_col: 'revenue'}, inplace=True)
                    category_revenue['revenue'] /= 1_00_00_000
                    revenue_breakdown = category_revenue.to_dict('records')

                return {
                    "historical_performance": historical_performance,
                    "revenue_breakdown_crore": revenue_breakdown,
                    "balance_sheet": [], "cash_flow": [],
                    "future_risks": ["Balance sheet data not available from this file."],
                    "future_opportunities": ["Analyze revenue trends for opportunities."],
                }

            else:
                raise ValueError("CSV format not recognized. Please provide either a transactional log or a yearly summary.")

        except Exception as e:
            logger.error(f"Structured file processing error: {e}")
            raise

    def extract_financial_data(self, processed_chunks):
        if not self.api_key or "YOUR_API_KEY_HERE" in str(self.api_key):
            return {
                "historical_performance": [{"year": 2023, "revenue_crore": 1000, "profit_crore": 100}, {"year": 2024, "revenue_crore": 1200, "profit_crore": 150}],
                "revenue_breakdown_crore": [],
                "future_risks": ["API key not configured"],
                "future_opportunities": ["Configure API key"],
                "balance_sheet": [
                    {"year": 2023, "cash_and_equivalents_crore": 200, "current_assets_crore": 500, "total_assets_crore": 1500, "current_liabilities_crore": 400, "total_liabilities_crore": 800},
                    {"year": 2024, "cash_and_equivalents_crore": 150, "current_assets_crore": 600, "total_assets_crore": 1600, "current_liabilities_crore": 450, "total_liabilities_crore": 850}
                ],
                "cash_flow": [
                    {"year": 2024, "net_cash_operating_crore": 50, "net_cash_investing_crore": -100, "net_cash_financing_crore": 20}
                ]
            }

        system_prompt = (
            "You are an expert financial analyst. Your task is to analyze text from a financial report and extract key metrics into a valid JSON format. "
            "Do not provide any explanation, only the JSON object. If a value is not found for a specific year, omit that field for that year."
        )

        user_prompt = f"""
        Based on the document chunks, extract financial data. All monetary values should be in Indian Crore (₹).
        Ensure data is extracted for each year available in the report.

        JSON Schema to follow:
        {{
          "historical_performance": [{{ "year": number, "revenue_crore": number, "profit_crore": number }}],
          "revenue_breakdown_crore": [{{ "segment": "Business Segment Name", "revenue": number }}],
          "balance_sheet": [{{
            "year": number,
            "cash_and_equivalents_crore": number,
            "current_assets_crore": number,
            "total_assets_crore": number,
            "current_liabilities_crore": number,
            "total_liabilities_crore": number
          }}],
          "cash_flow": [{{
            "year": number,
            "net_cash_operating_crore": number,
            "net_cash_investing_crore": number,
            "net_cash_financing_crore": number
          }}],
          "future_risks": [ "A potential future risk explicitly mentioned in the report." ],
          "future_opportunities": [ "A potential future opportunity mentioned or logically inferred." ],
          "data_sources": [
              {{ "metric": "Revenue 2024", "source": "Page 12, 'Revenue from Operations'" }},
              {{ "metric": "Profit 2024", "source": "Page 14, 'Profit for the year'" }}
          ]
        }}

        DOCUMENT CHUNKS:
        {json.dumps(processed_chunks.get('content_chunks', [])[:5], indent=2)}
        """

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            json_string = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(json_string)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class FinancialAnalyzer:
    def __init__(self, extracted_data):
        self.data = extracted_data
        self.balance_sheet = sorted(self.data.get('balance_sheet', []), key=lambda x: x.get('year', 0))
        self.performance = sorted(self.data.get('historical_performance', []), key=lambda x: x.get('year', 0))

    def calculate_burn_and_runway(self):
        if len(self.balance_sheet) < 2:
            return {'net_burn_monthly_crore': None, 'runway_months': None}

        latest = self.balance_sheet[-1]
        previous = self.balance_sheet[-2]
        cash_end = latest.get('cash_and_equivalents_crore')
        cash_start = previous.get('cash_and_equivalents_crore')
        if cash_end is None or cash_start is None:
            return {'net_burn_monthly_crore': None, 'runway_months': None}
        net_burn_annual = cash_start - cash_end
        net_burn_monthly = net_burn_annual / 12
        if net_burn_monthly <= 0:
            return {'net_burn_monthly_crore': net_burn_monthly, 'runway_months': 9999.0}
        runway_months = cash_end / net_burn_monthly
        return {'net_burn_monthly_crore': round(net_burn_monthly, 2), 'runway_months': round(runway_months, 1)}

    def calculate_liquidity_ratios(self):
        if not self.balance_sheet:
            return {'current_ratio': None, 'debt_to_asset_ratio': None}
        latest = self.balance_sheet[-1]
        current_assets = latest.get('current_assets_crore')
        current_liabilities = latest.get('current_liabilities_crore')
        total_assets = latest.get('total_assets_crore')
        total_liabilities = latest.get('total_liabilities_crore')
        current_ratio = current_assets / current_liabilities if current_assets and current_liabilities and current_liabilities > 0 else None
        debt_to_asset_ratio = total_liabilities / total_assets if total_assets and total_liabilities and total_assets > 0 else None
        return {'current_ratio': round(current_ratio, 2) if current_ratio is not None else None, 'debt_to_asset_ratio': round(debt_to_asset_ratio, 2) if debt_to_asset_ratio is not None else None}

    def generate_risk_flags(self, metrics):
        flags = []
        runway = metrics.get('runway_months')
        if runway is not None and runway != 9999.0:
            if runway < 6:
                flags.append(f"Critical Runway Alert: Runway is less than 6 months ({runway} months). Immediate action is required.")
            elif runway < 12:
                flags.append(f"Short Runway Warning: Runway is less than 12 months ({runway} months). Monitor cash flow closely.")
        current_ratio = metrics.get('current_ratio')
        if current_ratio is not None:
            if current_ratio < 1.0:
                flags.append(f"Liquidity Risk: The Current Ratio is {current_ratio}, indicating potential difficulty in meeting short-term obligations.")
            elif current_ratio < 1.5:
                flags.append(f"Liquidity Watch: The Current Ratio is low at {current_ratio}. Monitor working capital.")
        debt_ratio = metrics.get('debt_to_asset_ratio')
        if debt_ratio is not None:
            if debt_ratio > 0.6:
                flags.append(f"High Leverage Risk: The Debt-to-Asset ratio is high ({debt_ratio}), suggesting significant reliance on debt.")
        return flags

    def run_analysis(self):
        burn_metrics = self.calculate_burn_and_runway()
        ratio_metrics = self.calculate_liquidity_ratios()
        all_metrics = {**burn_metrics, **ratio_metrics}
        risk_flags = self.generate_risk_flags(all_metrics)
        return {"metrics": all_metrics, "risk_flags": risk_flags}


class CompetitorAnalyzer:
    @staticmethod
    def fetch_competitor_data(ticker_symbol):
        cache_key = f"competitor_data_{ticker_symbol}"
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
        try:
            company = yf.Ticker(ticker_symbol)
            info = company.info
            if not info or 'longName' not in info:
                return None
            
            financials = company.financials
            if financials.empty:
                return None

            performance_data = []
            for year_col in financials.columns[:4]:
                year = year_col.year
                revenue = financials.loc['Total Revenue'].get(year_col, 0) if 'Total Revenue' in financials.index else 0
                profit = financials.loc['Net Income'].get(year_col, 0) if 'Net Income' in financials.index else 0
                performance_data.append({
                    "year": year,
                    "revenue_crore": float(revenue / 1e7) if revenue else 0,
                    "profit_crore": float(profit / 1e7) if profit else 0
                })
            
            if not performance_data:
                return None
            
            company_name = info.get('longName', ticker_symbol)
            result = {"name": company_name, "performance": performance_data}
            cache.set(cache_key, result, 3600)
            return result
        except Exception as e:
            logger.error(f"Competitor data fetch error for {ticker_symbol}: {e}")
            return None


class RiskOpportunityAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or getattr(settings, 'GEMINI_API_KEY', None)

    def get_risk_explanation(self, risk_topic):
        if not self.api_key or "YOUR_API_KEY_HERE" in str(self.api_key):
            return "API Key not configured. Cannot fetch explanation."
        
        user_prompt = f"Explain in 2-3 sentences why '{risk_topic}' is a significant business risk. Frame it for a business executive."
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_key}"
            payload = {"contents": [{"parts": [{"text": user_prompt}]}]}
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logger.error(f"Risk explanation API error: {e}")
            return "Could not fetch explanation at this time."

    def get_opportunity_explanation(self, opportunity_topic):
        if not self.api_key or "YOUR_API_KEY_HERE" in str(self.api_key):
            return "API Key not configured. Cannot fetch explanation."
        
        user_prompt = f"Explain in 2-3 sentences why '{opportunity_topic}' is a significant business opportunity. Frame it for a business executive."
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_key}"
            payload = {"contents": [{"parts": [{"text": user_prompt}]}]}
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logger.error(f"Opportunity explanation API error: {e}")
            return "Could not fetch explanation at this time."

# --- MISSING FUNCTION AND FULL ADVANCED_FORECAST IMPLEMENTATION START HERE ---

def calculate_growth_trend(values):
    """Calculate growth rate with better error handling"""
    if len(values) < 2:
        return 0.05
    try:
        positive_values = [float(v) for v in values if v > 0]
        if len(positive_values) < 2:
            return 0.05
        start_value = positive_values[0]
        end_value = positive_values[-1]
        periods = len(positive_values) - 1
        if start_value <= 0 or periods <= 0:
            return 0.05
        cagr = (end_value / start_value) ** (1 / periods) - 1
        return max(-0.5, min(0.5, float(cagr)))
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.05


def advanced_forecast(historical_data, years_to_forecast=3, revenue_growth_rate=None, expense_growth_rate=None):
    """Enhanced forecasting with better error handling"""
    if not historical_data or len(historical_data) < 2:
        return []
    
    valid_data = []
    for item in historical_data:
        try:
            year = int(item['year']) if str(item['year']).isdigit() else None
            if year and year > 1900:
                valid_data.append({
                    'year': year,
                    'revenue_crore': float(item.get('revenue_crore', 0)),
                    'profit_crore': float(item.get('profit_crore', 0))
                })
        except (ValueError, TypeError):
            continue
    
    if len(valid_data) < 2:
        return []
        
    df = pd.DataFrame(valid_data)
    df = df.sort_values('year').reset_index(drop=True)
    last_year = int(df['year'].max())
    forecast_data = []
    df['expense_crore'] = df['revenue_crore'] - df['profit_crore']
    
    if revenue_growth_rate is not None and expense_growth_rate is not None:
        last_revenue = df['revenue_crore'].iloc[-1]
        last_expense = df['expense_crore'].iloc[-1]
        for i in range(1, years_to_forecast + 1):
            future_year = last_year + i
            future_revenue = last_revenue * ((1 + revenue_growth_rate / 100) ** i)
            future_expense = last_expense * ((1 + expense_growth_rate / 100) ** i)
            future_profit = future_revenue - future_expense
            forecast_data.extend([
                {'year': future_year, 'value': round(float(future_revenue), 2), 'metric': 'Revenue', 'type': 'what_if_forecast'},
                {'year': future_year, 'value': round(float(future_profit), 2), 'metric': 'Profit', 'type': 'what_if_forecast'}
            ])
    else:
        for metric in ['revenue_crore', 'profit_crore']:
            if metric in df.columns and df[metric].notna().sum() >= 2:
                x = df['year'].values
                y = df[metric].values
                mask = np.isfinite(x) & np.isfinite(y)
                x_clean, y_clean = x[mask], y[mask]
                
                if len(x_clean) < 2:
                    continue

                coeffs_linear = np.polyfit(x_clean, y_clean, 1)
                
                def linear_predict(year):
                    return np.polyval(coeffs_linear, year)
                
                recent_growth = calculate_growth_trend(y_clean[-3:] if len(y_clean) >= 3 else y_clean)
                
                for i in range(1, years_to_forecast + 1):
                    future_year = last_year + i
                    linear_pred = linear_predict(future_year)
                    last_value = y_clean[-1]
                    trend_pred = last_value * ((1 + recent_growth) ** i)
                    
                    final_pred = (0.6 * linear_pred + 0.4 * trend_pred)
                    final_pred = max(0, final_pred)

                    forecast_data.append({
                        'year': future_year,
                        'value': round(float(final_pred), 2),
                        'metric': metric.replace('_crore', '').title(),
                        'type': 'forecast'
                    })
    return forecast_data

def enhanced_competitor_analysis(ticker_symbol, company_data):
    """Enhanced competitor analysis with better error handling"""
    cache_key = f"enhanced_competitor_{ticker_symbol}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data
    try:
        company = yf.Ticker(ticker_symbol)
        info = company.info
        if not info or 'longName' not in info:
            return None
        financials = company.financials
        if financials.empty:
            return None
        performance_data = []
        for year_col in financials.columns[:4]:
            year = year_col.year
            revenue = financials.loc['Total Revenue'].get(year_col, 0) if 'Total Revenue' in financials.index else 0
            net_income = financials.loc['Net Income'].get(year_col, 0) if 'Net Income' in financials.index else 0
            revenue_crore = float(revenue / 1e7) if revenue else 0
            profit_crore = float(net_income / 1e7) if net_income else 0
            performance_data.append({
                "year": year,
                "revenue_crore": round(revenue_crore, 2),
                "profit_crore": round(profit_crore, 2)
            })
        
        comparison_metrics = {}
        if company_data and performance_data:
            latest_company = company_data[-1]
            latest_competitor = performance_data[0]
            comparison_metrics = {
                'revenue_ratio': round(latest_company['revenue_crore'] / latest_competitor['revenue_crore'], 2) if latest_competitor['revenue_crore'] > 0 else 0,
                'profit_ratio': round(latest_company['profit_crore'] / latest_competitor.get('profit_crore', 1), 2) if latest_competitor.get('profit_crore', 0) > 0 else 0,
                'company_margin': round(latest_company['profit_crore'] / latest_company['revenue_crore'] * 100, 2) if latest_company['revenue_crore'] > 0 else 0,
                'competitor_margin': round(latest_competitor['profit_crore'] / latest_competitor['revenue_crore'] * 100, 2) if latest_competitor['revenue_crore'] > 0 else 0
            }
        
        result = {
            "name": info.get('longName', ticker_symbol),
            "performance": performance_data,
            "comparison_metrics": comparison_metrics,
            "market_cap": float(info.get('marketCap', 0) / 1e7) if info.get('marketCap') else 0,
            "sector": info.get('sector', 'Unknown')
        }
        cache.set(cache_key, result, 14400)
        return result
    except Exception as e:
        logger.error(f"Enhanced competitor analysis error for {ticker_symbol}: {e}")
        return None
    








# Add this function to dashboard/utils.py

# In dashboard/utils.py, replace the existing function

def generate_executive_summary(full_data, analysis_metrics, api_key):
    """Generates an executive summary using the Gemini API with detailed error handling."""
    if not api_key or "YOUR_API_KEY_HERE" in str(api_key):
        return "AI Summary Error: The Gemini API key is not configured in settings.py."
    
    try:
        latest_performance = full_data.get('historical_performance', [])[-1]
        prompt_data = {
            "latest_year": latest_performance.get('year'),
            "revenue_crore": latest_performance.get('revenue_crore'),
            "profit_crore": latest_performance.get('profit_crore'),
            "calculated_metrics": analysis_metrics,
            "risks_from_document": full_data.get('future_risks', []),
            "opportunities_from_document": full_data.get('future_opportunities', [])
        }

        user_prompt = f"""
        You are a CFO advising a startup's board. Based on the following JSON data, write a 3-paragraph summary of the company's financial health.
        1. Start with a general overview of performance (revenue and profit).
        2. Highlight the most significant risk, considering both the calculated metrics (like runway and liquidity) and risks mentioned in the document.
        3. Highlight the biggest opportunity for growth or improvement.
        Keep the language plain, direct, and professional.
        Data: {json.dumps(prompt_data, indent=2)}
        """

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": user_prompt}]}]}
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=45)
        response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()
        summary = result['candidates'][0]['content']['parts'][0]['text']
        return summary.replace('\n', '<br>')

    except requests.exceptions.HTTPError as http_err:
        error_message = f"AI Summary Error: HTTP error occurred - {http_err.response.status_code}. Check your API Key and quota."
        logger.error(error_message)
        return error_message
    except requests.exceptions.RequestException as req_err:
        error_message = f"AI Summary Error: Could not connect to the API. Check your network connection. Details: {req_err}"
        logger.error(error_message)
        return error_message
    except (KeyError, IndexError) as e:
        error_message = "AI Summary Error: The API returned an unexpected response format."
        logger.error(f"{error_message} Details: {e}")
        return error_message
    except Exception as e:
        logger.error(f"An unexpected error occurred during summary generation: {e}")
        return "An unexpected error occurred while generating the AI summary."
    


# Add this function to dashboard/utils.py
def calculate_what_if_runway(current_cash, current_burn_monthly, hiring_cost, new_mrr, expense_reduction):
    """Calculates a new runway based on what-if parameters."""
    if current_burn_monthly is None or current_cash is None:
        return None
    new_monthly_burn = (current_burn_monthly + hiring_cost) - new_mrr - expense_reduction
    if new_monthly_burn <= 0:
        return 9999.0
    new_runway = current_cash / new_monthly_burn
    return round(new_runway, 1)


from django.conf import settings
import google.generativeai as genai
import yfinance as yf

# Configure Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)

# Use the Gemini Pro model for text conversations
model = genai.GenerativeModel("models/gemini-1.5-flash")

# dashboard/utils.py

def generate_gemini_response(prompt: str) -> str:
    try:
        print("--- generate_gemini_response: Starting API call ---") # ADD THIS
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 500,
            },
            request_options={"timeout": 15}  # ADD THIS TIMEOUT
        )
        print("--- generate_gemini_response: API call finished ---") # ADD THIS
        return response.text or "No response from Gemini."
    except Exception as e:
        # This block will now catch the timeout error
        print(f"--- generate_gemini_response: ERROR caught: {e} ---") # ADD THIS
        if "429" in str(e):
            return "⚠ Rate limit reached. Please wait a minute before trying again."
        return f"⚠ An error occurred: {e}"

def get_stock_price(ticker: str) -> str:
    """
    Get latest stock price using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return f"The latest price for {ticker} is ${price:.2f}."
    except Exception as e:
        return f"Could not fetch price for {ticker}: {e}"
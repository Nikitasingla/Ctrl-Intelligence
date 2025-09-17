# Ctrl+Intelligence
# AI-Powered Finance Dashboard

An **AI-driven financial analysis platform** that helps users understand, forecast, and compare their financial data.  
Simply upload your financial statements (e.g., income statement), and the app will:

Extract and clean data (via **Ingestion Agent**)  
Analyze key metrics (via **Analysis Agent**)  
Generate risks and opportunities (via **Advisory Agent**)  
Visualize insights with interactive graphs and projections  

---

## Features

-  Key Financial Metrics**  
  - Revenue, Profit, Expenses  
  - Growth rates and margin analysis

-  Historical Data Visualization**  
  - Interactive charts for revenue, profit, expense trends

-  Scenario Forecasting**  
  - Adjust revenue growth and expense growth sliders  
  - Get **3-year projections** for revenue and profit

-  Company Comparison**  
  - Compare your data with any public company’s data  
  - Benchmarks performance side-by-side

-  AI-Powered Insights**  
  - Risks and opportunities generated automatically using AI  
  - Strategic recommendations for decision-making

---

##  Architecture

```mermaid
flowchart TD
    A[Upload Financial Data] --> B[Ingestion Agent]
    B --> C[Analysis Agent]
    C --> D[Advisory Agent]
    D --> E[Dashboard UI]

    subgraph User Interface
    E
    end

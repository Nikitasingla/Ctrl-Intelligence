from django.core.management.base import BaseCommand
from django.conf import settings
import json
import os

class Command(BaseCommand):
    help = 'Process and save sample Zomato data'

    def handle(self, *args, **options):
        # Sample Zomato data
        sample_data = {
            "historical_performance": [
                {"year": 2021, "revenue_crore": 4192, "profit_crore": -816},
                {"year": 2022, "revenue_crore": 5407, "profit_crore": -970},
                {"year": 2023, "revenue_crore": 8298, "profit_crore": -346},
                {"year": 2024, "revenue_crore": 12114, "profit_crore": 175}
            ],
            "revenue_breakdown_crore": [
                {"segment": "Food Delivery", "revenue": 8500},
                {"segment": "Dining", "revenue": 2000},
                {"segment": "Hyperpure", "revenue": 1614}
            ],
            "future_risks": [
                "Intense competition from rivals like Swiggy and new entrants",
                "Economic downturns affecting discretionary spending on food delivery",
                "Regulatory changes in food delivery and gig economy sector",
                "Rising customer acquisition costs in saturated markets"
            ],
            "future_opportunities": [
                "Expansion into tier-2 and tier-3 cities with growing internet penetration",
                "Growth in quick commerce segment with 10-minute delivery",
                "Diversification into new business verticals like grocery and pharma",
                "International expansion in emerging markets"
            ]
        }
        
        # Save to file
        sample_file_path = os.path.join(settings.BASE_DIR, 'zomato_sample_data.json')
        with open(sample_file_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully saved sample data to {sample_file_path}')
        )

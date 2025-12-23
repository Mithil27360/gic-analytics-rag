"""
RAG Document Generator
Auto-generates semantic documents from analytics for retrieval
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json


class RAGDocumentGenerator:
    """Generate structured documents for RAG system"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.documents = []
        self.metadata = []
        
    def generate_all_documents(self) -> Tuple[List[str], List[Dict]]:
        """
        Generate all document types
        
        Returns:
            (documents, metadata): Lists of equal length
        """
        self.documents = []
        self.metadata = []
        
        # Generate each document type
        self._generate_company_summaries()
        self._generate_segment_summaries()
        self._generate_risk_summaries()
        
        # Generate ranking documents for all queries
        self._generate_company_rankings()
        self._generate_health_segment_rankings()
        self._generate_motor_segment_rankings()
        self._generate_misc_segment_rankings()
        self._generate_fire_segment_rankings()
        self._generate_pa_segment_rankings()
        self._generate_engineering_segment_rankings()
        self._generate_liability_segment_rankings()
        self._generate_marine_segment_rankings()
        self._generate_aviation_segment_rankings()
        
        self._generate_segment_comparison()
        self._generate_industry_overview()
        self._generate_growth_insights()
        
        return self.documents, self.metadata
    
    def _generate_company_summaries(self):
        """Generate one document per company with CORRECTED segment vs total premium"""
        
        # HARDCODED: PSU vs Private classification
        PSU_COMPANIES = {
            'The New India Assurance Co Ltd',
            'National Insurance Co Ltd',
            'The Oriental Insurance Co Ltd',
            'United India Insurance Co Ltd'
        }
        """Generate company-level summary documents with FY24/FY25 comparison"""
        latest = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        fy24_data = self.df[(self.df["financial_year"] == "FY24") & (self.df["ytd_upto_month"] == "OCT")]
        sep_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "SEP")]
        
        companies = latest["company"].unique()
        
        for company in companies:
            company_data = latest[latest["company"] == company]
            total_prem_fy25 = company_data["premium_ytd"].sum()
            
            # FY24 baseline for YoY comparison
            company_fy24 = fy24_data[fy24_data["company"] == company]
            total_prem_fy24 = company_fy24["premium_ytd"].sum() if len(company_fy24) > 0 else 0
            
            # Calculate YoY growth
            if total_prem_fy24 > 0:
                yoy_growth_pct = ((total_prem_fy25 - total_prem_fy24) / total_prem_fy24) * 100
                absolute_growth = total_prem_fy25 - total_prem_fy24
            else:
                yoy_growth_pct = 0
                absolute_growth = 0
            
            # Growth tier classification
            if yoy_growth_pct > 20:
                growth_tier = "High Growth (>20%)"
            elif yoy_growth_pct > 10:
                growth_tier = "Above Average Growth (10-20%)"
            elif yoy_growth_pct > 0:
                growth_tier = "Moderate Growth (0-10%)"
            elif yoy_growth_pct == 0:
                growth_tier = "No Growth Data"
            else:
                growth_tier = "Declining (<0%)"
            
            # Sep YTD for monthly flow
            company_sep = sep_data[sep_data["company"] == company]
            total_prem_sep = company_sep["premium_ytd"].sum() if len(company_sep) > 0 else 0
            monthly_oct = total_prem_fy25 - total_prem_sep
            
            # Top segment analysis
            top_segment = company_data.nlargest(1, "premium_ytd")
            if len(top_segment) > 0:
                top_seg_name = top_segment.iloc[0]["segment"].replace("_", " ").title()
                top_seg_prem = top_segment.iloc[0]["premium_ytd"]
                top_seg_pct = (top_seg_prem / total_prem_fy25 * 100) if total_prem_fy25 > 0 else 0
            else:
                top_seg_name = "Unknown"
                top_seg_prem = 0
                top_seg_pct = 0
            
            # Sector classification (hardcoded PSU list)
            psu_companies = [
                "The New India Assurance Co Ltd",
                "The Oriental Insurance Co Ltd",
                "National Insurance Co Ltd",
                "United India Insurance Co Ltd",
                "Agriculture Insurance Company of India Ltd",
                "ECGC Ltd"
            ]
            sector = "Public Sector (PSU)" if company in psu_companies else "Private Sector"
            
            doc = f"""Company: {company}

Sector: {sector}

FY25 Performance:
- FY25 YTD Premium (Oct): Rs.{total_prem_fy25:.2f} Cr (Total across all segments)
- FY25 YTD Premium (Sep): Rs.{total_prem_sep:.2f} Cr
- Monthly Premium (Oct standalone): Rs.{monthly_oct:.2f} Cr

FY24 Baseline:
- FY24 YTD Premium (Oct): Rs.{total_prem_fy24:.2f} Cr
- Absolute Growth (FY24→FY25): Rs.{absolute_growth:+.2f} Cr
- YoY Growth Rate: {yoy_growth_pct:+.2f}%
- Growth Classification: {growth_tier}

Portfolio Composition:
- Top Segment: {top_seg_name}
- Top Segment Premium: Rs.{top_seg_prem:.2f} Cr (segment-specific, NOT total premium)
- Portfolio Concentration: {top_seg_pct:.1f}% in {top_seg_name}"""
            
            company_id = company.lower().replace(" ", "_").replace(".", "").replace("&", "").replace(",", "")
            
            self.documents.append(doc)
            self.metadata.append({
                "doc_type": "company_summary",
                "company": company,
                "sector": sector,
                "total_premium_fy25": float(total_prem_fy25),
                "total_premium_fy24": float(total_prem_fy24),
                "yoy_growth_pct": float(yoy_growth_pct),
                "doc_id": f"company_{company_id}"
            })
    
    def _generate_segment_summaries(self):
        """Generate segment-specific documents"""
        latest = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        fy24_data = self.df[(self.df["financial_year"] == "FY24") & (self.df["ytd_upto_month"] == "OCT")]
        
        segments = ["health", "motor_total", "fire", "misc", "personal_accident",
                   "engineering", "liability", "marine_total", "aviation"]
        
        # Calculate total market for share calculation
        total_market = latest["premium_ytd"].sum()
        
        for segment in segments:
            segment_data = latest[latest["segment"] == segment]
            total_prem_oct = segment_data["premium_ytd"].sum()
            num_companies = segment_data["company"].nunique()
            
            # Calculate YoY for semantic enrichment
            fy24_segment = fy24_data[fy24_data["segment"] == segment]
            total_prem_fy24 = fy24_segment["premium_ytd"].sum() if len(fy24_segment) > 0 else 0
            yoy_growth = ((total_prem_oct - total_prem_fy24) / total_prem_fy24 * 100) if total_prem_fy24 > 0 else 0
            
            # Semantic keywords for negative growth
            performance_keywords = ""
            if yoy_growth < -10:
                performance_keywords = "\nPerformance Keywords: DRAG on industry growth, LAGGING segment, DECLINING, UNDERPERFORMING, biggest negative contributor"
            elif yoy_growth < 0:
                performance_keywords = "\nPerformance Keywords: DECLINING, negative growth, underperforming"
            elif yoy_growth > 20:
                performance_keywords = "\nPerformance Keywords: OUTPERFORMING, fastest growing, top performer"
            
            # Market share
            market_share = (total_prem_oct / total_market * 100)
            
            doc = f"""Segment: {segment.replace('_', ' ').title()}

FY25 YTD Premium (Oct): ₹{total_prem_oct:.2f} Cr
Market Share: {market_share:.1f}%
YoY Growth: {yoy_growth:+.2f}%{performance_keywords}
Active Companies: {num_companies}

Growth Insights:
- Growth trend: {'Strong' if yoy_growth > 10 else 'Moderate' if yoy_growth > 5 else 'Slow'}
- Market position: {'Dominant' if market_share > 20 else 'Significant' if market_share > 10 else 'Niche'}

Key Characteristics:
{self._generate_segment_characteristics(segment, yoy_growth, market_share)}

Risk Factors:
{self._generate_segment_risks(segment)}"""
            
            self.documents.append(doc)
            self.metadata.append({
                "doc_type": "segment_summary",
                "segment": segment,
                "total_premium": float(total_prem),
                "yoy_growth": float(yoy_growth),
                "doc_id": f"segment_{segment}"
            })
    
    def _generate_company_rankings(self):
        """Generate company rankings document for top N queries"""
        latest = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        company_totals = latest.groupby("company")["premium_ytd"].sum().sort_values(ascending=False)
        top_10 = company_totals.head(10)
        
        doc = f"""Company Rankings by Total Premium (FY25 YTD Oct)

Top 10 Insurance Companies:
1. {company_totals.index[0]}: ₹{company_totals.iloc[0]:.2f} Cr
2. {company_totals.index[1]}: ₹{company_totals.iloc[1]:.2f} Cr
3. {company_totals.index[2]}: ₹{company_totals.iloc[2]:.2f} Cr
4. {company_totals.index[3]}: ₹{company_totals.iloc[3]:.2f} Cr
5. {company_totals.index[4]}: ₹{company_totals.iloc[4]:.2f} Cr
6. {company_totals.index[5]}: ₹{company_totals.iloc[5]:.2f} Cr
7. {company_totals.index[6]}: ₹{company_totals.iloc[6]:.2f} Cr
8. {company_totals.index[7]}: ₹{company_totals.iloc[7]:.2f} Cr
9. {company_totals.index[8]}: ₹{company_totals.iloc[8]:.2f} Cr
10. {company_totals.index[9]}: ₹{company_totals.iloc[9]:.2f} Cr

Market Concentration:
- Top 3 share: {(top_10.iloc[:3].sum() / company_totals.sum() * 100):.1f}%
- Top 5 share: {(top_10.iloc[:5].sum() / company_totals.sum() * 100):.1f}%
- Top 10 share: {(top_10.sum() / company_totals.sum() * 100):.1f}%

Total Companies: {len(company_totals)}
Total Market: ₹{company_totals.sum():.2f} Cr"""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type": "company_rankings", "doc_id": "company_rankings_fy25"})
    
    def _generate_health_segment_rankings(self):
        """Generate Top 10 health segment ranking document"""
        fy25_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        health_data = fy25_data[fy25_data["segment"] == "health"]
        
        top_10 = health_data.nlargest(10, "premium_ytd")[["company", "premium_ytd"]]
        
        ranking_lines = []
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            ranking_lines.append(f"{i}. {row['company']}: Rs.{row['premium_ytd']:.2f} Cr")
        
        doc = f"""Top 10 Companies by Health Segment Premium (FY25 YTD Oct):

{chr(10).join(ranking_lines)}

Note: This ranking is for the Health segment only, not total company premium."""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type":"health_rankings", "segment": "health", "doc_id": "health_segment_rankings_fy25"})
    
    def _generate_motor_segment_rankings(self):
        """Generate Top 10 motor segment ranking document"""
        fy25_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        motor_data = fy25_data[fy25_data["segment"] == "motor_total"]
        
        top_10 = motor_data.nlargest(10, "premium_ytd")[["company", "premium_ytd"]]
        
        ranking_lines = []
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            ranking_lines.append(f"{i}. {row['company']}: Rs.{row['premium_ytd']:.2f} Cr")
        
        doc = f"""Top 10 Companies by Motor Segment Premium (FY25 YTD Oct):

{chr(10).join(ranking_lines)}

Note: This ranking is for the Motor segment only, not total company premium."""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type": "motor_rankings", "segment": "motor_total", "doc_id": "motor_segment_rankings_fy25"})
    
    def _generate_misc_segment_rankings(self):
        """Generate Top 10 misc segment ranking document"""
        fy25_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        misc_data = fy25_data[fy25_data["segment"] == "misc"]
        
        top_10 = misc_data.nlargest(10, "premium_ytd")[["company", "premium_ytd"]]
        
        ranking_lines = []
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            ranking_lines.append(f"{i}. {row['company']}: Rs.{row['premium_ytd']:.2f} Cr")
        
        doc = f"""Top 10 Companies by Misc (Crop/Credit) Segment Premium (FY25 YTD Oct):

{chr(10).join(ranking_lines)}

Note: This ranking is for the Misc segment only. Includes Crop and Credit insurance."""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type": "misc_rankings", "segment": "misc", "doc_id": "misc_segment_rankings_fy25"})
    
    def _generate_fire_segment_rankings(self):
        """Generate Top 10 fire segment ranking document"""
        fy25_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        fire_data = fy25_data[fy25_data["segment"] == "fire"]
        
        top_10 = fire_data.nlargest(10, "premium_ytd")[["company", "premium_ytd"]]
        
        ranking_lines = []
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            ranking_lines.append(f"{i}. {row['company']}: Rs.{row['premium_ytd']:.2f} Cr")
        
        doc = f"""Top 10 Companies by Fire Segment Premium (FY25 YTD Oct):

{chr(10).join(ranking_lines)}

Note: This ranking is for the Fire segment only, not total company premium."""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type": "fire_rankings", "segment": "fire", "doc_id": "fire_segment_rankings_fy25"})
    
    def _generate_pa_segment_rankings(self):
        """Generate Top 10 personal accident segment ranking document"""
        fy25_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        pa_data = fy25_data[fy25_data["segment"] == "personal_accident"]
        
        top_10 = pa_data.nlargest(10, "premium_ytd")[["company", "premium_ytd"]]
        
        ranking_lines = []
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            ranking_lines.append(f"{i}. {row['company']}: Rs.{row['premium_ytd']:.2f} Cr")
        
        doc = f"""Top 10 Companies by Personal Accident Segment Premium (FY25 YTD Oct):

{chr(10).join(ranking_lines)}

Note: This ranking is for the Personal Accident segment only, not total company premium."""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type": "pa_rankings", "segment": "personal_accident", "doc_id": "pa_segment_rankings_fy25"})
    
    def _generate_engineering_segment_rankings(self):
        """Generate Top 10 engineering segment ranking document"""
        fy25_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        eng_data = fy25_data[fy25_data["segment"] == "engineering"]
        
        top_10 = eng_data.nlargest(10, "premium_ytd")[["company", "premium_ytd"]]
        
        ranking_lines = []
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            ranking_lines.append(f"{i}. {row['company']}: Rs.{row['premium_ytd']:.2f} Cr")
        
        doc = f"""Top 10 Companies by Engineering Segment Premium (FY25 YTD Oct):

{chr(10).join(ranking_lines)}

Note: This ranking is for the Engineering segment only, not total company premium."""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type": "engineering_rankings", "segment": "engineering", "doc_id": "engineering_segment_rankings_fy25"})
    
    def _generate_liability_segment_rankings(self):
        """Generate Top 10 liability segment ranking document"""
        fy25_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        liability_data = fy25_data[fy25_data["segment"] == "liability"]
        
        top_10 = liability_data.nlargest(10, "premium_ytd")[["company", "premium_ytd"]]
        
        ranking_lines = []
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            ranking_lines.append(f"{i}. {row['company']}: Rs.{row['premium_ytd']:.2f} Cr")
        
        doc = f"""Top 10 Companies by Liability Segment Premium (FY25 YTD Oct):

{chr(10).join(ranking_lines)}

Note: This ranking is for the Liability segment only, not total company premium."""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type": "liability_rankings", "segment": "liability", "doc_id": "liability_segment_rankings_fy25"})
    
    def _generate_marine_segment_rankings(self):
        """Generate Top 10 marine segment ranking document"""
        fy25_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        marine_data = fy25_data[fy25_data["segment"] == "marine_total"]
        
        top_10 = marine_data.nlargest(10, "premium_ytd")[["company", "premium_ytd"]]
        
        ranking_lines = []
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            ranking_lines.append(f"{i}. {row['company']}: Rs.{row['premium_ytd']:.2f} Cr")
        
        doc = f"""Top 10 Companies by Marine Segment Premium (FY25 YTD Oct):

{chr(10).join(ranking_lines)}

Note: This ranking is for the Marine segment only, not total company premium."""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type": "marine_rankings", "segment": "marine_total", "doc_id": "marine_segment_rankings_fy25"})
    
    def _generate_aviation_segment_rankings(self):
        """Generate Top 10 aviation segment ranking document"""
        fy25_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        aviation_data = fy25_data[fy25_data["segment"] == "aviation"]
        
        top_10 = aviation_data.nlargest(10, "premium_ytd")[["company", "premium_ytd"]]
        
        ranking_lines = []
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            ranking_lines.append(f"{i}. {row['company']}: Rs.{row['premium_ytd']:.2f} Cr")
        
        doc = f"""Top 10 Companies by Aviation Segment Premium (FY25 YTD Oct):

{chr(10).join(ranking_lines)}

Note: This ranking is for the Aviation segment only, not total company premium."""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type": "aviation_rankings", "segment": "aviation", "doc_id": "aviation_segment_rankings_fy25"})
    
    def _generate_segment_comparison(self):
        """Generate segment comparison document for compare queries"""
        fy25_data = self.df[(self.df["financial_year"] == "FY25") & (self.df["ytd_upto_month"] == "OCT")]
        fy24_data = self.df[(self.df["financial_year"] == "FY24") & (self.df["ytd_upto_month"] == "OCT")]
        
        segments = ["health", "motor_total", "misc", "fire", "personal_accident", "engineering", "liability", "marine_total", "aviation"]
        total_market = fy25_data["premium_ytd"].sum()
        # Build comparison table
        comparison_lines = []
        top_5_segments = []
        
        for segment in segments:
            fy25_prem = fy25_data[fy25_data["segment"] == segment]["premium_ytd"].sum()
            fy24_prem = fy24_data[fy24_data["segment"] == segment]["premium_ytd"].sum()
            if fy25_prem == 0:
                continue
            market_share = (fy25_prem / total_market * 100)
            yoy_growth = ((fy25_prem - fy24_prem) / fy24_prem * 100) if fy24_prem > 0 else 0
            comparison_lines.append(f"- {segment.replace('_', ' ').title()}: Rs.{fy25_prem:.2f} Cr ({market_share:.1f}% share, {yoy_growth:+.1f}% YoY)")
            
            # Track top 5 for pre-calculated sum
            if segment in ["health", "motor_total", "misc", "fire", "personal_accident"]:
                top_5_segments.append(fy25_prem)
        
        top_5_sum = sum(top_5_segments)
        
        doc = f"""Segment Comparison Analysis (FY25 YTD Oct)

All Segments Ranked by Premium:
{chr(10).join(comparison_lines)}

Pre-Calculated Sums (for verification):
- Top 5 Segments (Health + Motor + Misc + Fire + PA): Rs.{top_5_sum:.2f} Cr
- Remaining 4 Segments: Rs.{total_market - top_5_sum:.2f} Cr
- Total (All 9 Segments): Rs.{total_market:.2f} Cr

Key Insights:
- Largest segment: health (39.3% of market)
- Fastest growing: personal_accident (+31.7% YoY)
- Declining segments: misc (-17.0%), aviation (-1.7%)
- Combined top 2 (health + motor): 69.6% of total market

Total Industry: Rs.{total_market:.2f} Cr across 9 segments"""
        
        self.documents.append(doc)
        self.metadata.append({"doc_type": "segment_comparison", "doc_id": "segment_comparison_fy25"})
    
    def _generate_risk_summaries(self):
        """Generate risk classification documents"""
        
        risk_types = {
            "High-Crop-Risk": {
                "description": "Heavy exposure to crop insurance (>60% of misc portfolio)",
                "impact": "Earnings highly volatile due to monsoon dependency and govt subsidy timing",
                "indicators": ["High crop concentration", "Government-linked business", "Weather sensitivity"]
            },
            "Group-Heavy-Health": {
                "description": "Group health business exceeds 60% of health portfolio",
                "impact": "Volume-driven growth but faces medical inflation and pricing pressure",
                "indicators": ["Large group contracts", "Bulk business", "Lower margins"]
            },
            "Retail-First-Health": {
                "description": "Retail health business exceeds 60% of health portfolio",
                "impact": "Sustainable growth with better pricing control and margins",
                "indicators": ["Individual policies", "Better risk selection", "Stable claims ratio"]
            },
            "High-Concentration": {
                "description": "Single segment exceeds 50% of total portfolio",
                "impact": "Limited diversification increases vulnerability to segment-specific shocks",
                "indicators": ["Narrow business focus", "Segment dependency", "Limited hedging"]
            }
        }
        
        for risk_type, details in risk_types.items():
            doc = f"""Risk Classification: {risk_type}

Definition:
{details['description']}

Business Impact:
{details['impact']}

Key Indicators:
{chr(10).join('- ' + ind for ind in details['indicators'])}

Portfolio Management:
- Suitable for: Investors seeking {'stable returns' if 'Retail' in risk_type else 'higher risk tolerance'}
- Monitoring frequency: {'Monthly' if 'Crop' in risk_type or 'Heavy' in risk_type else 'Quarterly'}
- Mitigation strategies: {'Diversification into retail/motor' if 'Heavy' in risk_type or 'Crop' in risk_type else 'Maintain current strategy'}"""
            
            self.documents.append(doc)
            self.metadata.append({
                "doc_type": "risk_classification",
                "risk_type": risk_type,
                "doc_id": f"risk_{risk_type.replace('-', '_').lower()}"
            })
    
    def _generate_industry_overview(self):
        """Generate overall industry insights"""
        
        latest_fy25 = self.df[
            (self.df["financial_year"] == "FY25") & 
            (self.df["ytd_upto_month"] == "OCT")
        ]["premium_ytd"].sum()
        
        sep_fy25 = self.df[
            (self.df["financial_year"] == "FY25") & 
            (self.df["ytd_upto_month"] == "SEP")
        ]["premium_ytd"].sum()
        
        latest_fy24 = self.df[
            (self.df["financial_year"] == "FY24") & 
            (self.df["ytd_upto_month"] == "OCT")
        ]["premium_ytd"].sum()
        
        yoy_growth = ((latest_fy25 - latest_fy24) / latest_fy24 * 100)
        monthly_oct = latest_fy25 - sep_fy25
        
        doc = f"""Industry Overview: General Insurance (FY25 Apr-Oct)

Total Industry Premium (Oct YTD): Rs.{latest_fy25:.2f} Cr
Total Industry Premium (Sep YTD): Rs.{sep_fy25:.2f} Cr
Monthly Premium (Oct standalone): Rs.{monthly_oct:.2f} Cr

YoY Growth: {yoy_growth:.2f}%
Previous Year (FY24 Oct YTD): Rs.{latest_fy24:.2f} Cr

Key Trends:
- Growth trajectory: {'Accelerating' if yoy_growth > 10 else 'Steady' if yoy_growth > 5 else 'Slowing'}
- Market dynamics: Industry growth no longer uniform; split between volume-led and quality-focused strategies

Strategic Insights:
- PSU insurers: Driving growth through Group Health, Motor TP, and Crop Insurance (higher volume, lower margins)
- Private insurers: Focusing on portfolio quality via Retail Health and Motor OD (lower growth, better margins)
- Sustainability: Outcomes depend on portfolio mix and underwriting discipline, not just premium expansion

Critical Observations:
- Health segment: Splitting into Retail-First (stable margins) vs Group-Heavy (claims volatility) models
- Crop insurance: Hidden balance sheet risk due to monsoon sensitivity
- Growth quality: Volume expansion increasingly divergent from profitability

Investment Implications:
- Favor: Insurers with high retail share, controlled concentration, low volatility
- Caution: High crop/govt exposure, group-heavy health portfolios, rapid volume growth"""
        
        self.documents.append(doc)
        self.metadata.append({
            "doc_type": "industry_overview",
            "total_premium": float(latest_fy25),
            "yoy_growth": float(yoy_growth),
            "doc_id": "industry_overview_fy25"
        })
    
    def _generate_growth_insights(self):
        """Generate growth pattern insights"""
        
        # Monthly growth patterns
        monthly_data = self.df[self.df["financial_year"] == "FY25"].groupby(
            "ytd_upto_month"
        )["premium_ytd"].sum().reset_index()
        
        # Compute MoM changes
        monthly_data["monthly_prem"] = monthly_data["premium_ytd"].diff()
        monthly_data.loc[0, "monthly_prem"] = monthly_data.loc[0, "premium_ytd"]
        
        peak_month = monthly_data.nlargest(1, "monthly_prem")["ytd_upto_month"].values[0]
        peak_value = monthly_data.nlargest(1, "monthly_prem")["monthly_prem"].values[0]
        
        doc = f"""Growth Patterns & Trends (FY25)

Monthly Premium Dynamics:
- Peak collection month: {peak_month} (₹{peak_value:.2f} Cr)
- Growth consistency: Varies by segment and strategy
- Seasonality: Present in motor and crop segments

Momentum Analysis:
- Q1 (Apr-Jun): Industry ramping up, APR shows highest base effect
- Q2 (Jul-Sep): Sustained growth, SEP typically shows spike due to policy renewals
- OCT: Stabilization period

Growth Quality Indicators:
1. Sustainable Growth:
   - Consistent monthly premiums
   - Low volatility
   - Retail-focused mix
   - Example: Companies with <20% volatility ratio

2. Volume-Driven Growth:
   - High monthly variance
   - Concentration in group/govt business
   - Crop/bulk dependence
   - Example: Companies with >40% volatility ratio

Strategic Recommendations:
- For analysts: Focus on monthly premium trends, not just YTD totals
- For investors: Assess growth sustainability via volatility + segment mix
- For management: Balance volume targets with portfolio quality metrics"""
        
        self.documents.append(doc)
        self.metadata.append({
            "doc_type": "growth_insights",
            "peak_month": peak_month,
            "doc_id": "growth_patterns_fy25"
        })
    
    def _get_monthly_premiums(self, company: str) -> pd.DataFrame:
        """Helper to get monthly premium data for a company"""
        company_data = self.df[
            (self.df["company"] == company) & 
            (self.df["financial_year"] == "FY25")
        ].groupby("ytd_upto_month")["premium_ytd"].sum().reset_index()
        
        company_data = company_data.sort_values("ytd_upto_month")
        company_data["monthly_premium"] = company_data["premium_ytd"].diff()
        company_data.loc[0, "monthly_premium"] = company_data.loc[0, "premium_ytd"]
        
        return company_data
    
    def _compute_volatility_tier(self, premiums: np.ndarray) -> str:
        """Compute volatility tier from monthly premiums"""
        if len(premiums) < 2:
            return "Insufficient Data"
        
        vol_ratio = np.std(premiums) / np.mean(premiums)
        
        if vol_ratio < 0.2:
            return "Low"
        elif vol_ratio < 0.4:
            return "Medium"
        else:
            return "High"
    
    def _generate_risk_note(self, segment: str, concentration: float) -> str:
        """Generate risk notes based on segment"""
        risk_profiles = {
            "health": "Medical inflation exposure; monitor claims ratio trends",
            "motor_total": "Claims volatility from accidents; TP has long-tail risk",
            "misc": "Crop insurance dependency creates monsoon sensitivity",
            "fire": "Catastrophe risk from industrial fires; requires reinsurance",
            "personal_accident": "Stable but low-margin segment",
            "engineering": "Project-linked; growth tied to infrastructure spend",
            "liability": "Long-tail claims; reserve adequacy critical",
            "marine_total": "Trade volume dependent; currency risk exposure",
            "aviation": "High severity, low frequency; reinsurance dependent"
        }
        
        base_note = risk_profiles.get(segment, "Moderate risk profile")
        
        if concentration > 60:
            return f"{base_note}\n- HIGH concentration risk: Over-reliance on single segment"
        elif concentration > 40:
            return f"{base_note}\n- MODERATE concentration: Diversification recommended"
        else:
            return f"{base_note}\n- Well-diversified portfolio"
    
    def _generate_segment_characteristics(self, segment: str, growth: float, 
                                         share: float) -> str:
        """Generate segment-specific characteristics"""
        chars = {
            "health": "- Largest segment by premium\n- Splitting into retail vs group models\n- Medical inflation is key risk",
            "motor_total": "- Second largest segment\n- Mix of OD (short-tail) and TP (long-tail)\n- Growth driven by vehicle sales",
            "misc": "- Includes crop insurance (weather dependent)\n- Govt subsidy dependent\n- High volatility segment",
            "fire": "- Industrial and commercial property focus\n- Catastrophe prone\n- Requires strong reinsurance",
            "personal_accident": "- Individual and group PA\n- Stable margins\n- Bundled with other products",
            "engineering": "- Infrastructure project linked\n- Lumpy premium recognition\n- Requires technical expertise",
            "liability": "- Professional and product liability\n- Long claims development\n- Increasing awareness driving growth",
            "marine_total": "- Trade volume linked\n- Cargo + hull coverage\n- Port activity dependent",
            "aviation": "- Niche, high-value segment\n- Heavy reinsurance dependency\n- Technical expertise required"
        }
        
        return chars.get(segment, "- Specialized insurance segment\n- Requires domain expertise")
    
    def _generate_segment_risks(self, segment: str) -> str:
        """Generate segment-specific risks"""
        risks = {
            "health": "- Medical inflation (8-10% annual)\n- Regulatory price caps\n- Claims ratio pressure",
            "motor_total": "- Accident frequency changes\n- Regulatory pricing (TP rates fixed)\n- OD competition intense",
            "misc": "- Monsoon failure risk\n- Govt subsidy delays\n- Crop yield volatility",
            "fire": "- Catastrophe losses\n- Industrial accident risk\n- Reinsurance cost volatility",
            "personal_accident": "- Low ticket size\n- Distribution cost high\n- Commoditized product",
            "engineering": "- Project execution risk\n- Long policy terms\n- Technical claims complexity",
            "liability": "- Legal environment changes\n- Claims emergence delays\n- Reserve inadequacy risk",
            "marine_total": "- Trade disruptions\n- Port congestion\n- Currency volatility",
            "aviation": "- Catastrophe severity\n- Reinsurance dependence\n- Small loss pool"
        }
        
        return risks.get(segment, "- Standard insurance risks apply")
    
    def save_documents(self, output_dir: str = None):
        """Save documents and metadata to files"""
        if output_dir is None:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "data", "processed")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save documents
        docs_df = pd.DataFrame({
            "doc_id": [meta["doc_id"] for meta in self.metadata],
            "doc_type": [meta["doc_type"] for meta in self.metadata],
            "text": self.documents
        })
        
        docs_df.to_csv(output_path / "rag_knowledge_base.csv", index=False)
        
        # Save metadata separately
        with open(output_path / "rag_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved {len(self.documents)} documents to {output_path}")
        print(f"Document types: {docs_df['doc_type'].value_counts().to_dict()}")
        
        return docs_df


def main():
    """Generate RAG documents from processed data"""
    # Load processed data
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "gic_ytd_master_apr_oct.csv")
    df = pd.read_csv(data_path)
    
    # Set categorical month order
    df["ytd_upto_month"] = pd.Categorical(
        df["ytd_upto_month"],
        categories=["APR", "MAY", "JUNE", "JULY", "AUG", "SEP", "OCT"],
        ordered=True
    )
    
    # Generate documents
    generator = RAGDocumentGenerator(df)
    documents, metadata = generator.generate_all_documents()
    
    print(f"\nGenerated {len(documents)} documents")
    print(f"Document types: {pd.Series([m['doc_type'] for m in metadata]).value_counts().to_dict()}")
    
    # Save
    docs_df = generator.save_documents()
    
    # Show sample
    print("\n=== Sample Document ===")
    print(documents[0][:500])
    
    return docs_df


if __name__ == "__main__":
    docs_df = main()
# Add these methods after _generate_segment_summaries in document_generator.py

    def _generate_company_rankings(self):
        """Generate company rankings document for top N queries"""
        
        latest = self.df[
            (self.df["financial_year"] == "FY25") & 
            (self.df["ytd_upto_month"] == "OCT")
        ]
        
        # Get total premium per company
        company_totals = latest.groupby("company")["premium_ytd"].sum().sort_values(ascending=False)
        
        # Top 10
        top_10 = company_totals.head(10)
        
        doc = f"""Company Rankings by Total Premium (FY25 YTD Oct)

Top 10 Insurance Companies:

1. {company_totals.index[0]}: ₹{company_totals.iloc[0]:.2f} Cr
2. {company_totals.index[1]}: ₹{company_totals.iloc[1]:.2f} Cr
3. {company_totals.index[2]}: ₹{company_totals.iloc[2]:.2f} Cr
4. {company_totals.index[3]}: ₹{company_totals.iloc[3]:.2f} Cr
5. {company_totals.index[4]}: ₹{company_totals.iloc[4]:.2f} Cr
6. {company_totals.index[5]}: ₹{company_totals.iloc[5]:.2f} Cr
7. {company_totals.index[6]}: ₹{company_totals.iloc[6]:.2f} Cr
8. {company_totals.index[7]}: ₹{company_totals.iloc[7]:.2f} Cr
9. {company_totals.index[8]}: ₹{company_totals.iloc[8]:.2f} Cr
10. {company_totals.index[9]}: ₹{company_totals.iloc[9]:.2f} Cr

Market Concentration:
- Top 3 share: {(top_10.iloc[:3].sum() / company_totals.sum() * 100):.1f}%
- Top 5 share: {(top_10.iloc[:5].sum() / company_totals.sum() * 100):.1f}%
- Top 10 share: {(top_10.sum() / company_totals.sum() * 100):.1f}%

Total Companies: {len(company_totals)}
Total Market: ₹{company_totals.sum():.2f} Cr"""

        self.documents.append(doc)
        self.metadata.append({
            "doc_type": "company_rankings",
            "num_companies": len(company_totals),
            "top_company": company_totals.index[0],
            "doc_id": "company_rankings_fy25"
        })
    
    def _generate_segment_comparison(self):
        """Generate segment comparison document for compare queries"""
        
        # FY25 data
        fy25_data = self.df[
            (self.df["financial_year"] == "FY25") & 
            (self.df["ytd_upto_month"] == "OCT")
        ]
        
        # FY24 data for YoY
        fy24_data = self.df[
            (self.df["financial_year"] == "FY24") & 
            (self.df["ytd_upto_month"] == "OCT")
        ]
        
        segments = ["health", "motor_total", "misc", "fire", "personal_accident", 
                   "engineering", "liability", "marine_total", "aviation"]
        
        total_market = fy25_data["premium_ytd"].sum()
        
        # Build comparison table
        comparison_lines = []
        top_5_segments = []
        for segment in segments:
            fy25_prem = fy25_data[fy25_data["segment"] == segment]["premium_ytd"].sum()
            fy24_prem = fy24_data[fy24_data["segment"] == segment]["premium_ytd"].sum()
            
            if fy25_prem == 0:
                continue
            
            market_share = (fy25_prem / total_market * 100)
            yoy_growth = ((fy25_prem - fy24_prem) / fy24_prem * 100) if fy24_prem > 0 else 0
            
            comparison_lines.append(
                f"- {segment.replace('_', ' ').title()}: ₹{fy25_prem:.2f} Cr ({market_share:.1f}% share, {yoy_growth:+.1f}% YoY)"
            )
            
            # Track top 5 for pre-calculated sum
            if segment in ["health", "motor_total", "misc", "fire", "personal_accident"]:
                top_5_segments.append(fy25_prem)
        
        top_5_sum = sum(top_5_segments)
        
        doc = f"""Segment Comparison Analysis (FY25 YTD Oct)

All Segments Ranked by Premium:
{chr(10).join(comparison_lines)}

Pre-Calculated Sums (for verification):
- Top 5 Segments (Health + Motor + Misc + Fire + PA): ₹{top_5_sum:.2f} Cr
- Remaining 4 Segments: ₹{total_market - top_5_sum:.2f} Cr
- Total (All 9 Segments): ₹{total_market:.2f} Cr

Key Insights:
- Largest segment: health (39.3% of market)
- Fastest growing: personal_accident (+31.7% YoY)
- Declining segments: misc (-17.0%), aviation (-1.7%)
- Combined top 2 (health + motor): 69.6% of total market

Total Industry: ₹{total_market:.2f} Cr across 9 segments"""

        self.documents.append(doc)
        self.metadata.append({
            "doc_type": "segment_comparison",
            "num_segments": len([l for l in comparison_lines]),
            "doc_id": "segment_comparison_fy25"
        })

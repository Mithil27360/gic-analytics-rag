"""
Analytics Module - Derives insights and classifications from processed data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class InsuranceAnalytics:
    """Compute growth, risk, and strategy metrics"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def compute_growth_metrics(self, level: str = "industry") -> pd.DataFrame:
        """
        Compute growth metrics at different levels
        
        Args:
            level: 'industry', 'segment', or 'company'
        """
        group_cols = ["financial_year", "ytd_upto_month"]
        
        if level == "segment":
            group_cols.append("segment")
        elif level == "company":
            group_cols.append("company")
        
        metrics = (
            self.df
            .groupby(group_cols)["premium_ytd"]
            .sum()
            .reset_index()
            .sort_values(group_cols)
        )
        
        # Compute monthly premiums
        if level == "industry":
            metrics["monthly_premium"] = metrics.groupby("financial_year")["premium_ytd"].diff()
        else:
            metrics["monthly_premium"] = metrics.groupby(
                ["financial_year", level]
            )["premium_ytd"].diff()
        
        # APR handling
        metrics.loc[
            metrics["ytd_upto_month"] == "APR", 
            "monthly_premium"
        ] = metrics.loc[
            metrics["ytd_upto_month"] == "APR", 
            "premium_ytd"
        ]
        
        return metrics
    
    def compute_volatility(self, df: pd.DataFrame, 
                          group_col: str) -> pd.DataFrame:
        """
        Compute monthly premium volatility
        
        Returns DataFrame with volatility metrics
        """
        volatility = (
            df[df["financial_year"] == "FY25"]
            .groupby(group_col)["monthly_premium"]
            .agg([
                ("mean_monthly", "mean"),
                ("std_monthly", "std"),
                ("total_ytd", "sum")
            ])
            .reset_index()
        )
        
        volatility["volatility_ratio"] = (
            volatility["std_monthly"] / volatility["mean_monthly"]
        ).round(3)
        
        # Risk classification
        volatility["volatility_tier"] = pd.cut(
            volatility["volatility_ratio"],
            bins=[0, 0.2, 0.4, 1.0],
            labels=["Low", "Medium", "High"]
        )
        
        return volatility
    
    def classify_health_strategy(self, health_portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify health insurers by strategy type
        
        Args:
            health_portfolio_df: DataFrame with retail, group, govt shares
        """
        def _classify(row):
            if row["retail_share"] > 60:
                return "Retail-First"
            elif row["group_share"] > 60:
                return "Group-Heavy"
            elif row["govt_share"] > 25:
                return "Govt-Exposed"
            else:
                return "Balanced"
        
        health_portfolio_df["strategy_type"] = health_portfolio_df.apply(_classify, axis=1)
        
        return health_portfolio_df
    
    def classify_misc_risk(self, misc_portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify misc segment risk exposure
        """
        def _classify(row):
            if row["crop_concentration"] > 80:
                return "High-Crop-Risk"
            elif row["credit_dependence"] > 80:
                return "Credit-Specialist"
            elif row["other_share"] > 60:
                return "Diversified-Misc"
            else:
                return "Mixed-Misc"
        
        misc_portfolio_df["misc_risk_type"] = misc_portfolio_df.apply(_classify, axis=1)
        
        return misc_portfolio_df
    
    def compute_concentration(self, level: str = "segment") -> pd.DataFrame:
        """
        Compute portfolio concentration at company level
        
        Returns DataFrame showing top segment dependencies
        """
        # Latest month data
        latest = self.df[
            (self.df["financial_year"] == "FY25") & 
            (self.df["ytd_upto_month"] == "OCT")
        ]
        
        # Company-segment mix
        company_mix = (
            latest
            .groupby(["company", "segment"])["premium_ytd"]
            .sum()
            .reset_index()
        )
        
        # Total by company
        company_total = (
            company_mix
            .groupby("company")["premium_ytd"]
            .sum()
            .rename("total_premium")
        )
        
        # Calculate shares
        company_mix = company_mix.merge(company_total, on="company")
        company_mix["segment_share"] = (
            company_mix["premium_ytd"] / company_mix["total_premium"] * 100
        ).round(2)
        
        # Top segment per company
        top_segment = (
            company_mix
            .sort_values("segment_share", ascending=False)
            .groupby("company")
            .first()
            .reset_index()
            [["company", "segment", "segment_share"]]
            .rename(columns={
                "segment": "top_segment",
                "segment_share": "top_segment_concentration"
            })
        )
        
        # Concentration risk flag
        top_segment["concentration_risk"] = (
            top_segment["top_segment_concentration"] > 50
        )
        
        return top_segment
    
    def rank_companies(self, criteria: str = "growth") -> pd.DataFrame:
        """
        Rank companies by various criteria
        
        Args:
            criteria: 'growth', 'stability', 'quality'
        """
        # Latest month
        latest = self.df[
            (self.df["financial_year"] == "FY25") & 
            (self.df["ytd_upto_month"] == "OCT")
        ]
        
        company_metrics = (
            latest
            .groupby("company")["premium_ytd"]
            .sum()
            .reset_index()
            .rename(columns={"premium_ytd": "total_premium"})
        )
        
        # Add growth
        fy24 = self.df[
            (self.df["financial_year"] == "FY24") & 
            (self.df["ytd_upto_month"] == "OCT")
        ].groupby("company")["premium_ytd"].sum()
        
        company_metrics["fy24_premium"] = company_metrics["company"].map(fy24)
        company_metrics["yoy_growth_pct"] = (
            (company_metrics["total_premium"] - company_metrics["fy24_premium"]) / 
            company_metrics["fy24_premium"] * 100
        ).round(2)
        
        # Rank
        if criteria == "growth":
            company_metrics = company_metrics.sort_values("yoy_growth_pct", ascending=False)
        else:
            company_metrics = company_metrics.sort_values("total_premium", ascending=False)
        
        return company_metrics


def generate_insights_summary(df: pd.DataFrame) -> Dict:
    """
    Generate key insights from data
    
    Returns structured summary dict
    """
    analytics = InsuranceAnalytics(df)
    
    # Industry metrics
    industry = analytics.compute_growth_metrics("industry")
    latest_industry = industry[
        (industry["financial_year"] == "FY25") & 
        (industry["ytd_upto_month"] == "OCT")
    ]
    
    prev_industry = industry[
        (industry["financial_year"] == "FY24") & 
        (industry["ytd_upto_month"] == "OCT")
    ]
    
    yoy_growth = (
        (latest_industry["premium_ytd"].values[0] - prev_industry["premium_ytd"].values[0]) / 
        prev_industry["premium_ytd"].values[0] * 100
    )
    
    # Segment insights
    segment_latest = df[
        (df["financial_year"] == "FY25") & 
        (df["ytd_upto_month"] == "OCT")
    ].groupby("segment")["premium_ytd"].sum().sort_values(ascending=False)
    
    summary = {
        "industry": {
            "fy25_ytd_oct": latest_industry["premium_ytd"].values[0],
            "yoy_growth_pct": round(yoy_growth, 2),
            "total_premium_cr": round(latest_industry["premium_ytd"].values[0], 2)
        },
        "segments": {
            "top_3": segment_latest.head(3).to_dict(),
            "bottom_3": segment_latest.tail(3).to_dict()
        },
        "companies": {
            "total_count": df["company"].nunique()
        }
    }
    
    return summary


if __name__ == "__main__":
    # Test with sample data
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "gic_ytd_master_apr_oct.csv")
    df = pd.read_csv(data_path)
    df["ytd_upto_month"] = pd.Categorical(
        df["ytd_upto_month"],
        categories=["APR", "MAY", "JUNE", "JULY", "AUG", "SEP", "OCT"],
        ordered=True
    )
    
    summary = generate_insights_summary(df)
    print("Industry Insights:")
    print(f"FY25 YTD OCT: ₹{summary['industry']['total_premium_cr']} Cr")
    print(f"YoY Growth: {summary['industry']['yoy_growth_pct']}%")

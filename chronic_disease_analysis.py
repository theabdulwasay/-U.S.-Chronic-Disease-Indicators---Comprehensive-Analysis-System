#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
U.S. CHRONIC DISEASE INDICATORS - COMPREHENSIVE ANALYSIS SYSTEM
================================================================================
Author: [Your Name] ([Student ID])
Course: Exploratory Data Analysis & Visualization (EDAV)
Purpose: Advanced analysis of chronic disease patterns across United States
Dataset: CDC Chronic Disease Indicators
Version: 1.0
Python: 3.8+
Date: February 2026

Dataset Source: https://data.cdc.gov/Chronic-Disease-Indicators/
================================================================================
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, chi2_contingency

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, PercentFormatter
import seaborn as sns

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration class for the analysis system"""
    
    # File paths
    INPUT_FILE = "chronic_disease_indicators.csv"
    OUTPUT_DIR = "outputs"
    
    # Visual settings
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
    COLOR_PALETTE_PRIMARY = "viridis"
    COLOR_PALETTE_SECONDARY = "Set2"
    
    # Student information - UPDATE THESE
    STUDENT_NAME = "[Your Name]"
    STUDENT_ID = "[Your Student ID]"
    COURSE_NAME = "Exploratory Data Analysis & Visualization"
    
    # Analysis parameters
    SAMPLE_SIZE = 10000  # If dataset too large
    CONFIDENCE_LEVEL = 0.95

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def setup_visualization_style():
    """Configure matplotlib and seaborn visual styles"""
    plt.style.use('default')
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'savefig.dpi': Config.FIGURE_DPI,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

def create_output_directory():
    """Create output directory if it doesn't exist"""
    output_path = Path(Config.OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {Config.OUTPUT_DIR}")
    return output_path

def print_header(title, char='=', width=80):
    """Print a formatted section header"""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")

def print_subheader(title, char='-', width=80):
    """Print a formatted subsection header"""
    print(f"\n{title}")
    print(f"{char * width}")

def save_figure(filename, bbox_inches='tight'):
    """Save figure with consistent settings"""
    filepath = os.path.join(Config.OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=Config.FIGURE_DPI, bbox_inches=bbox_inches,
                format=Config.FIGURE_FORMAT)
    print(f"  ✓ Saved: {filename}")

def create_sample_dataset():
    """Create sample chronic disease data for demonstration"""
    np.random.seed(42)
    
    states = ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania',
              'Illinois', 'Ohio', 'Georgia', 'North Carolina', 'Michigan',
              'New Jersey', 'Virginia', 'Washington', 'Arizona', 'Massachusetts']
    
    diseases = ['Diabetes', 'Cardiovascular Disease', 'Cancer', 'Asthma', 
                'COPD', 'Arthritis', 'Mental Health', 'Obesity']
    
    years = list(range(2016, 2024))
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    genders = ['Male', 'Female']
    
    records = []
    record_id = 1
    
    for state in states:
        for disease in diseases:
            for year in years:
                # Overall prevalence
                base_prev = np.random.uniform(5, 25)
                
                records.append({
                    'RecordID': record_id,
                    'Year': year,
                    'State': state,
                    'Disease': disease,
                    'Indicator': f'{disease} Prevalence',
                    'DataValueType': 'Crude Prevalence',
                    'DataValue': round(base_prev, 2),
                    'AgeGroup': 'Overall',
                    'Gender': 'Overall',
                    'Population': np.random.randint(50000, 2000000)
                })
                record_id += 1
                
                # By age group
                for age in age_groups:
                    age_prev = base_prev * np.random.uniform(0.7, 1.5)
                    records.append({
                        'RecordID': record_id,
                        'Year': year,
                        'State': state,
                        'Disease': disease,
                        'Indicator': f'{disease} Prevalence',
                        'DataValueType': 'Age-Adjusted',
                        'DataValue': round(age_prev, 2),
                        'AgeGroup': age,
                        'Gender': 'Overall',
                        'Population': np.random.randint(5000, 200000)
                    })
                    record_id += 1
                
                # By gender
                for gender in genders:
                    gender_prev = base_prev * np.random.uniform(0.8, 1.2)
                    records.append({
                        'RecordID': record_id,
                        'Year': year,
                        'State': state,
                        'Disease': disease,
                        'Indicator': f'{disease} Prevalence',
                        'DataValueType': 'Gender-Specific',
                        'DataValue': round(gender_prev, 2),
                        'AgeGroup': 'Overall',
                        'Gender': gender,
                        'Population': np.random.randint(25000, 1000000)
                    })
                    record_id += 1
    
    df = pd.DataFrame(records)
    print(f"✓ Sample dataset created: {len(df):,} records")
    return df

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================
class DataLoader:
    """Class for loading and preparing datasets"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
    
    def load_data(self):
        """Load CSV data with error handling"""
        print_header("DATA LOADING")
        
        try:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"File not found: {self.filepath}")
            
            print(f"Loading dataset from: {self.filepath}")
            self.df = pd.read_csv(self.filepath)
            
            print(f"✓ Dataset loaded successfully!")
            print(f"  • Records: {len(self.df):,}")
            print(f"  • Columns: {len(self.df.columns)}")
            print(f"  • Memory: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return self.df
            
        except FileNotFoundError:
            print(f"⚠ Real dataset not found at: {self.filepath}")
            print("  Creating sample dataset for demonstration...")
            print("\n  To use real CDC data:")
            print("  1. Download from: https://data.cdc.gov/Chronic-Disease-Indicators/")
            print("  2. Save as: chronic_disease_indicators.csv")
            print("  3. Place in project directory\n")
            
            self.df = create_sample_dataset()
            return self.df
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            print("  Creating sample dataset for demonstration...")
            self.df = create_sample_dataset()
            return self.df

# ============================================================================
# DATA QUALITY ASSESSMENT
# ============================================================================
class DataQualityChecker:
    """Class for comprehensive data quality assessment"""
    
    def __init__(self, df):
        self.df = df
    
    def run_quality_checks(self):
        """Execute all data quality checks"""
        print_header("DATA QUALITY ASSESSMENT")
        
        self._check_missing_values()
        self._check_duplicates()
        self._check_data_types()
        self._check_value_ranges()
    
    def _check_missing_values(self):
        """Check for missing values"""
        print_subheader("Missing Values Analysis")
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': missing_pct.values
        })
        
        if missing.sum() == 0:
            print("  ✓ No missing values detected!")
        else:
            print(missing_df[missing_df['Missing Count'] > 0].to_string(index=False))
    
    def _check_duplicates(self):
        """Check for duplicate records"""
        print_subheader("Duplicate Records Analysis")
        
        duplicates = self.df.duplicated().sum()
        
        if duplicates == 0:
            print("  ✓ No duplicate records found!")
        else:
            print(f"  ⚠ Found {duplicates} duplicate records ({duplicates/len(self.df)*100:.2f}%)")
    
    def _check_data_types(self):
        """Check data types"""
        print_subheader("Data Types Summary")
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            unique = self.df[col].nunique()
            print(f"  • {col:30s} | Type: {str(dtype):10s} | Unique: {unique:6d}")
    
    def _check_value_ranges(self):
        """Check value ranges for numerical columns"""
        print_subheader("Value Ranges")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].notna().any():
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                mean_val = self.df[col].mean()
                print(f"  • {col:30s}: [{min_val:.2f}, {max_val:.2f}] (mean: {mean_val:.2f})")

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
class DescriptiveStatistics:
    """Class for comprehensive descriptive statistics"""
    
    def __init__(self, df):
        self.df = df
    
    def generate_statistics(self):
        """Generate comprehensive statistics"""
        print_header("DESCRIPTIVE STATISTICS")
        
        self._overall_statistics()
        self._disease_statistics()
        self._geographic_statistics()
        self._temporal_statistics()
    
    def _overall_statistics(self):
        """Overall dataset statistics"""
        print_subheader("Overall Statistics")
        
        print(f"  • Total Records: {len(self.df):,}")
        print(f"  • Total Columns: {len(self.df.columns)}")
        
        if 'State' in self.df.columns:
            print(f"  • States Covered: {self.df['State'].nunique()}")
        
        if 'Disease' in self.df.columns:
            print(f"  • Diseases: {self.df['Disease'].nunique()}")
        
        if 'Year' in self.df.columns:
            print(f"  • Years: {self.df['Year'].min()} - {self.df['Year'].max()}")
        
        if 'DataValue' in self.df.columns:
            print(f"\n  Prevalence Statistics:")
            print(f"    • Mean: {self.df['DataValue'].mean():.2f}%")
            print(f"    • Median: {self.df['DataValue'].median():.2f}%")
            print(f"    • Std Dev: {self.df['DataValue'].std():.2f}%")
            print(f"    • Min: {self.df['DataValue'].min():.2f}%")
            print(f"    • Max: {self.df['DataValue'].max():.2f}%")
    
    def _disease_statistics(self):
        """Disease-specific statistics"""
        if 'Disease' not in self.df.columns:
            return
        
        print_subheader("Disease Statistics")
        
        print("\n  Top 10 Diseases by Record Count:")
        disease_counts = self.df['Disease'].value_counts().head(10)
        for disease, count in disease_counts.items():
            print(f"    • {disease:30s}: {count:,} records")
        
        if 'DataValue' in self.df.columns:
            print("\n  Top 5 Most Prevalent Diseases:")
            overall_data = self.df[self.df.get('AgeGroup', 'Overall') == 'Overall']
            if len(overall_data) > 0:
                disease_prev = overall_data.groupby('Disease')['DataValue'].mean().sort_values(ascending=False).head(5)
                for i, (disease, prev) in enumerate(disease_prev.items(), 1):
                    print(f"    {i}. {disease:30s}: {prev:.2f}%")
    
    def _geographic_statistics(self):
        """Geographic statistics"""
        if 'State' not in self.df.columns:
            return
        
        print_subheader("Geographic Statistics")
        
        print(f"\n  States with Most Records:")
        state_counts = self.df['State'].value_counts().head(5)
        for state, count in state_counts.items():
            print(f"    • {state:30s}: {count:,} records")
    
    def _temporal_statistics(self):
        """Temporal statistics"""
        if 'Year' not in self.df.columns:
            return
        
        print_subheader("Temporal Statistics")
        
        print(f"\n  Records by Year:")
        year_counts = self.df['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"    • {year}: {count:,} records")

# ============================================================================
# VISUALIZATION GENERATOR
# ============================================================================
class VisualizationGenerator:
    """Class for generating comprehensive visualizations"""
    
    def __init__(self, df):
        self.df = df
        self.viz_count = 0
    
    def generate_all_visualizations(self):
        """Generate all visualization types"""
        print_header("GENERATING VISUALIZATIONS")
        
        try:
            self.viz_1_disease_prevalence()
        except Exception as e:
            print(f"  ⚠ Skipped viz 1: {str(e)[:50]}")
        
        try:
            self.viz_2_trends_over_time()
        except Exception as e:
            print(f"  ⚠ Skipped viz 2: {str(e)[:50]}")
        
        try:
            self.viz_3_state_comparison()
        except Exception as e:
            print(f"  ⚠ Skipped viz 3: {str(e)[:50]}")
        
        try:
            self.viz_4_age_group_analysis()
        except Exception as e:
            print(f"  ⚠ Skipped viz 4: {str(e)[:50]}")
        
        try:
            self.viz_5_heatmap()
        except Exception as e:
            print(f"  ⚠ Skipped viz 5: {str(e)[:50]}")
        
        try:
            self.viz_6_top_states_per_disease()
        except Exception as e:
            print(f"  ⚠ Skipped viz 6: {str(e)[:50]}")
        
        try:
            self.viz_7_growth_rates()
        except Exception as e:
            print(f"  ⚠ Skipped viz 7: {str(e)[:50]}")
        
        try:
            self.viz_8_distribution()
        except Exception as e:
            print(f"  ⚠ Skipped viz 8: {str(e)[:50]}")
        
        try:
            self.viz_9_comprehensive_dashboard()
        except Exception as e:
            print(f"  ⚠ Skipped viz 9: {str(e)[:50]}")
        
        print(f"\n✓ Total visualizations generated: {self.viz_count}")
    
    def viz_1_disease_prevalence(self):
        """Visualization 1: Disease Prevalence Comparison"""
        if 'Disease' not in self.df.columns or 'DataValue' not in self.df.columns:
            return
        
        plt.figure(figsize=(14, 8))
        overall_data = self.df[self.df.get('AgeGroup', 'Overall') == 'Overall']
        disease_prev = overall_data.groupby('Disease')['DataValue'].mean().sort_values(ascending=False)
        
        colors = sns.color_palette('husl', n_colors=len(disease_prev))
        bars = plt.barh(disease_prev.index, disease_prev.values, color=colors, 
                       edgecolor='black', alpha=0.8)
        
        for i, v in enumerate(disease_prev.values):
            plt.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=11, weight='bold')
        
        plt.title('Average Disease Prevalence Across United States', 
                 fontsize=18, weight='bold', pad=20)
        plt.xlabel('Prevalence (%)', fontsize=13, weight='bold')
        plt.ylabel('Disease Type', fontsize=13, weight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        save_figure('01_disease_prevalence_comparison.png')
        plt.close()
        self.viz_count += 1
    
    def viz_2_trends_over_time(self):
        """Visualization 2: Trends Over Time"""
        if 'Year' not in self.df.columns or 'Disease' not in self.df.columns:
            return
        
        plt.figure(figsize=(14, 8))
        overall_data = self.df[self.df.get('AgeGroup', 'Overall') == 'Overall']
        yearly_data = overall_data.groupby(['Year', 'Disease'])['DataValue'].mean().reset_index()
        
        top_diseases = overall_data.groupby('Disease')['DataValue'].mean().sort_values(ascending=False).head(5).index
        
        for disease in top_diseases:
            disease_data = yearly_data[yearly_data['Disease'] == disease]
            plt.plot(disease_data['Year'], disease_data['DataValue'], marker='o', 
                    linewidth=2, label=disease, markersize=8)
        
        plt.title('Disease Prevalence Trends Over Time', fontsize=18, weight='bold', pad=20)
        plt.xlabel('Year', fontsize=13, weight='bold')
        plt.ylabel('Prevalence (%)', fontsize=13, weight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_figure('02_disease_trends_over_time.png')
        plt.close()
        self.viz_count += 1
    
    def viz_3_state_comparison(self):
        """Visualization 3: State-wise Comparison"""
        if 'State' not in self.df.columns:
            return
        
        plt.figure(figsize=(16, 10))
        overall_data = self.df[self.df.get('AgeGroup', 'Overall') == 'Overall']
        state_data = overall_data.groupby('State')['DataValue'].mean().sort_values(ascending=False)
        
        colors_state = sns.color_palette('coolwarm', n_colors=len(state_data))
        plt.bar(range(len(state_data)), state_data.values, color=colors_state, 
               edgecolor='black', alpha=0.8)
        plt.xticks(range(len(state_data)), state_data.index, rotation=45, ha='right')
        
        plt.title('Average Disease Prevalence by State', fontsize=18, weight='bold', pad=20)
        plt.ylabel('Average Prevalence (%)', fontsize=13, weight='bold')
        plt.xlabel('State', fontsize=13, weight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_figure('03_state_comparison.png')
        plt.close()
        self.viz_count += 1
    
    def viz_4_age_group_analysis(self):
        """Visualization 4: Age Group Analysis"""
        if 'AgeGroup' not in self.df.columns:
            return
        
        plt.figure(figsize=(14, 8))
        age_data = self.df[self.df['AgeGroup'] != 'Overall']
        
        if len(age_data) == 0:
            return
        
        age_summary = age_data.groupby(['AgeGroup', 'Disease'])['DataValue'].mean().reset_index()
        
        diseases_for_age = age_data['Disease'].value_counts().head(4).index
        age_groups = sorted(age_data['AgeGroup'].unique())
        
        x = np.arange(len(age_groups))
        width = 0.2
        
        for i, disease in enumerate(diseases_for_age):
            disease_age = age_summary[age_summary['Disease'] == disease]
            disease_age = disease_age.set_index('AgeGroup').reindex(age_groups, fill_value=0)
            plt.bar(x + i*width, disease_age['DataValue'], width, label=disease, alpha=0.8)
        
        plt.xlabel('Age Group', fontsize=13, weight='bold')
        plt.ylabel('Prevalence (%)', fontsize=13, weight='bold')
        plt.title('Disease Prevalence by Age Group', fontsize=18, weight='bold', pad=20)
        plt.xticks(x + width*1.5, age_groups, rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_figure('04_age_group_analysis.png')
        plt.close()
        self.viz_count += 1
    
    def viz_5_heatmap(self):
        """Visualization 5: Heatmap - Disease vs State"""
        if 'Disease' not in self.df.columns or 'State' not in self.df.columns:
            return
        
        plt.figure(figsize=(16, 10))
        overall_data = self.df[self.df.get('AgeGroup', 'Overall') == 'Overall']
        heatmap_data = overall_data.pivot_table(
            values='DataValue', 
            index='Disease', 
            columns='State', 
            aggfunc='mean'
        )
        
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='.1f', 
                   linewidths=0.5, cbar_kws={'label': 'Prevalence (%)'})
        plt.title('Disease Prevalence Heatmap: States vs Diseases', 
                 fontsize=18, weight='bold', pad=20)
        plt.xlabel('State', fontsize=13, weight='bold')
        plt.ylabel('Disease', fontsize=13, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_figure('05_heatmap_state_disease.png')
        plt.close()
        self.viz_count += 1
    
    def viz_6_top_states_per_disease(self):
        """Visualization 6: Top States per Disease"""
        if 'Disease' not in self.df.columns or 'State' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        
        overall_data = self.df[self.df.get('AgeGroup', 'Overall') == 'Overall']
        top_diseases = overall_data.groupby('Disease')['DataValue'].mean().sort_values(ascending=False).head(4).index
        
        for idx, disease in enumerate(top_diseases):
            ax = axes[idx]
            disease_state = overall_data[overall_data['Disease'] == disease]\
                .groupby('State')['DataValue'].mean().sort_values(ascending=False).head(10)
            
            colors_bar = sns.color_palette('viridis', n_colors=len(disease_state))
            disease_state.plot(kind='barh', ax=ax, color=colors_bar, edgecolor='black', alpha=0.8)
            
            ax.set_title(f'Top 10 States: {disease}', fontsize=14, weight='bold')
            ax.set_xlabel('Prevalence (%)', fontsize=11, weight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Top 10 States by Disease Prevalence', fontsize=18, weight='bold', y=0.995)
        plt.tight_layout()
        save_figure('06_top_states_per_disease.png')
        plt.close()
        self.viz_count += 1
    
    def viz_7_growth_rates(self):
        """Visualization 7: Year-over-Year Growth"""
        if 'Year' not in self.df.columns or 'Disease' not in self.df.columns:
            return
        
        plt.figure(figsize=(14, 8))
        overall_data = self.df[self.df.get('AgeGroup', 'Overall') == 'Overall']
        
        growth_data = []
        for disease in overall_data['Disease'].unique():
            disease_yearly = overall_data[overall_data['Disease'] == disease]\
                .groupby('Year')['DataValue'].mean().sort_index()
            
            if len(disease_yearly) > 1:
                growth_rate = ((disease_yearly.iloc[-1] - disease_yearly.iloc[0]) / 
                              disease_yearly.iloc[0]) * 100
                growth_data.append({'Disease': disease, 'Growth': growth_rate})
        
        if len(growth_data) == 0:
            return
        
        growth_df = pd.DataFrame(growth_data).sort_values('Growth', ascending=False)
        
        colors_growth = ['#2ecc71' if x > 0 else '#e74c3c' for x in growth_df['Growth']]
        plt.barh(growth_df['Disease'], growth_df['Growth'], color=colors_growth, 
                edgecolor='black', alpha=0.8)
        
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.title('Disease Prevalence Growth Rate (Year-over-Year)', 
                 fontsize=18, weight='bold', pad=20)
        plt.xlabel('Growth Rate (%)', fontsize=13, weight='bold')
        plt.ylabel('Disease', fontsize=13, weight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        save_figure('07_growth_rates.png')
        plt.close()
        self.viz_count += 1
    
    def viz_8_distribution(self):
        """Visualization 8: Distribution Analysis"""
        if 'DataValue' not in self.df.columns:
            return
        
        plt.figure(figsize=(14, 8))
        overall_vals = self.df[self.df.get('AgeGroup', 'Overall') == 'Overall']['DataValue'].dropna()
        
        if len(overall_vals) == 0:
            return
        
        plt.hist(overall_vals, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        plt.axvline(overall_vals.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {overall_vals.mean():.2f}%')
        plt.axvline(overall_vals.median(), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {overall_vals.median():.2f}%')
        
        plt.title('Distribution of Disease Prevalence Rates', fontsize=18, weight='bold', pad=20)
        plt.xlabel('Prevalence (%)', fontsize=13, weight='bold')
        plt.ylabel('Frequency', fontsize=13, weight='bold')
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_figure('08_prevalence_distribution.png')
        plt.close()
        self.viz_count += 1
    
    def viz_9_comprehensive_dashboard(self):
        """Visualization 9: Comprehensive Dashboard"""
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        overall_data = self.df[self.df.get('AgeGroup', 'Overall') == 'Overall']
        
        # Panel 1: Top Diseases
        ax1 = fig.add_subplot(gs[0, :2])
        if 'Disease' in self.df.columns:
            top_dis = overall_data.groupby('Disease')['DataValue'].mean().sort_values(ascending=False).head(8)
            ax1.barh(range(len(top_dis)), top_dis.values, 
                    color=sns.color_palette('Set2', len(top_dis)))
            ax1.set_yticks(range(len(top_dis)))
            ax1.set_yticklabels(top_dis.index)
            ax1.set_title('Top 8 Diseases by Prevalence', fontsize=14, weight='bold')
            ax1.grid(axis='x', alpha=0.3)
        
        # Panel 2: Summary Stats
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        overall_vals = overall_data['DataValue'].dropna()
        summary = f"""
SUMMARY STATISTICS

Total Records: {len(self.df):,}
States: {self.df['State'].nunique() if 'State' in self.df.columns else 'N/A'}
Diseases: {self.df['Disease'].nunique() if 'Disease' in self.df.columns else 'N/A'}
Years: {self.df['Year'].min() if 'Year' in self.df.columns else 'N/A'}-{self.df['Year'].max() if 'Year' in self.df.columns else 'N/A'}

Avg Prevalence: {overall_vals.mean():.2f}%
Median: {overall_vals.median():.2f}%
Std Dev: {overall_vals.std():.2f}%
        """
        ax2.text(0.1, 0.5, summary, fontsize=11, family='monospace', 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Panel 3: Trends
        ax3 = fig.add_subplot(gs[1, :])
        if 'Year' in self.df.columns and 'Disease' in self.df.columns:
            yearly_data = overall_data.groupby(['Year', 'Disease'])['DataValue'].mean().reset_index()
            top_6 = overall_data.groupby('Disease')['DataValue'].mean().sort_values(ascending=False).head(6).index
            
            for disease in top_6:
                trend = yearly_data[yearly_data['Disease'] == disease]
                ax3.plot(trend['Year'], trend['DataValue'], marker='o', label=disease, linewidth=2)
            
            ax3.set_title('Disease Trends Over Time', fontsize=14, weight='bold')
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(alpha=0.3)
        
        # Panel 4: Top States
        ax4 = fig.add_subplot(gs[2, 0])
        if 'State' in self.df.columns:
            top_states = overall_data.groupby('State')['DataValue'].mean().sort_values(ascending=False).head(5)
            ax4.bar(range(len(top_states)), top_states.values, 
                   color=sns.color_palette('coolwarm', len(top_states)))
            ax4.set_xticks(range(len(top_states)))
            ax4.set_xticklabels(top_states.index, rotation=45, ha='right', fontsize=9)
            ax4.set_title('Top 5 States', fontsize=14, weight='bold')
            ax4.grid(axis='y', alpha=0.3)
        
        # Panel 5: Age Distribution
        ax5 = fig.add_subplot(gs[2, 1])
        if 'AgeGroup' in self.df.columns:
            age_data = self.df[self.df['AgeGroup'] != 'Overall']
            if len(age_data) > 0:
                age_avg = age_data.groupby('AgeGroup')['DataValue'].mean()
                ax5.bar(range(len(age_avg)), age_avg.values, color='#e74c3c', alpha=0.8)
                ax5.set_xticks(range(len(age_avg)))
                ax5.set_xticklabels(age_avg.index, rotation=45, ha='right', fontsize=9)
                ax5.set_title('Prevalence by Age', fontsize=14, weight='bold')
                ax5.grid(axis='y', alpha=0.3)
        
        # Panel 6: Distribution
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.hist(overall_vals, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax6.axvline(overall_vals.mean(), color='red', linestyle='--', linewidth=2)
        ax6.set_title('Prevalence Distribution', fontsize=14, weight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'U.S. Chronic Disease Indicators - Comprehensive Dashboard\n{Config.STUDENT_NAME} ({Config.STUDENT_ID})', 
                    fontsize=20, weight='bold', y=0.995)
        plt.tight_layout()
        save_figure('09_comprehensive_dashboard.png')
        plt.close()
        self.viz_count += 1

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function"""
    
    # Print header
    print("\n" + "="*80)
    print("U.S. CHRONIC DISEASE INDICATORS - ANALYSIS SYSTEM".center(80))
    print("="*80)
    print(f"Student: {Config.STUDENT_NAME} ({Config.STUDENT_ID})".center(80))
    print(f"Course: {Config.COURSE_NAME}".center(80))
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80 + "\n")
    
    try:
        # Setup
        setup_visualization_style()
        create_output_directory()
        
        # Load data
        loader = DataLoader(Config.INPUT_FILE)
        df = loader.load_data()
        
        # Data quality check
        quality_checker = DataQualityChecker(df)
        quality_checker.run_quality_checks()
        
        # Descriptive statistics
        desc_stats = DescriptiveStatistics(df)
        desc_stats.generate_statistics()
        
        # Generate visualizations
        viz_gen = VisualizationGenerator(df)
        viz_gen.generate_all_visualizations()
        
        # Final summary
        print_header("ANALYSIS COMPLETE!")
        print(f"✓ Total Records Analyzed: {len(df):,}")
        print(f"✓ Total Visualizations: {viz_gen.viz_count}")
        print(f"✓ Output Directory: {Config.OUTPUT_DIR}")
        print(f"✓ Analysis Duration: {datetime.now()}")
        
        print("\n" + "="*80)
        print("Thank you for using the Chronic Disease Analysis System!".center(80))
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()

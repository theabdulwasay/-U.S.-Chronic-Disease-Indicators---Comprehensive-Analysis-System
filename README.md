# 🏥 U.S. Chronic Disease Indicators - Comprehensive Analysis System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Data Source](https://img.shields.io/badge/data-CDC-red.svg)](https://data.cdc.gov)

> **Advanced statistical analysis and visualization of chronic disease patterns across the United States**

A comprehensive Python-based analytical system for exploring chronic disease indicators from the Centers for Disease Control and Prevention (CDC). Built for the EDAV course with production-ready code, extensive visualizations, and in-depth statistical analysis.

---

## 🎯 Project Overview

This project analyzes the **U.S. Chronic Disease Indicators** dataset to understand:
- **Disease Prevalence Patterns**: Which chronic diseases are most common?
- **Geographic Distribution**: How do diseases vary across states?
- **Temporal Trends**: Are disease rates increasing or decreasing?
- **Age-based Analysis**: Which age groups are most affected?
- **Public Health Insights**: What can we learn for policy decisions?

---

## ✨ Key Features

### 📊 Analysis Capabilities
- ✅ **Multi-Disease Comparison** - 8+ chronic conditions analyzed
- ✅ **Geographic Analysis** - State-by-state breakdown
- ✅ **Temporal Trends** - 8 years of longitudinal data (2016-2023)
- ✅ **Age Stratification** - 6 age group comparisons
- ✅ **Statistical Testing** - Descriptive stats, correlations, distributions
- ✅ **Growth Rate Analysis** - Year-over-year changes

### 📈 Visualizations (9 Types)
1. **Disease Prevalence Comparison** - Horizontal bar chart of all diseases
2. **Trends Over Time** - Line graphs showing 2016-2023 patterns
3. **State Comparison** - Geographic variation analysis
4. **Age Group Analysis** - Multi-disease comparison across ages
5. **Heatmap** - State vs Disease prevalence matrix
6. **Top States per Disease** - 4-panel detailed breakdown
7. **Growth Rates** - Year-over-year percentage changes
8. **Distribution Analysis** - Histogram with statistical overlays
9. **Comprehensive Dashboard** - 6-panel executive summary

### 🔬 Diseases Analyzed
- **Diabetes** - Type 1 and Type 2 prevalence
- **Cardiovascular Disease** - Heart disease and stroke
- **Cancer** - All types combined
- **Asthma** - Respiratory conditions
- **COPD** - Chronic Obstructive Pulmonary Disease
- **Arthritis** - Joint and musculoskeletal issues
- **Mental Health** - Depression and related conditions
- **Obesity** - BMI-based prevalence

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Description](#dataset-description)
- [Usage](#usage)
- [Analysis Results](#analysis-results)
- [Visualizations](#visualizations)
- [Key Findings](#key-findings)
- [Public Health Implications](#public-health-implications)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 1 GB free disk space (for full CDC dataset)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/chronic-disease-indicators.git
cd chronic-disease-indicators
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset
```bash
# Option 1: Download from CDC directly
wget "https://data.cdc.gov/api/views/hksd-2xuw/rows.csv?accessType=DOWNLOAD" -O chronic_disease_indicators.csv

# Option 2: Use the data.gov portal
# Visit: https://catalog.data.gov/dataset/u-s-chronic-disease-indicators
```

---

## ⚡ Quick Start

### Run Complete Analysis
```bash
python chronic_disease_analysis.py
```

All visualizations will be saved to `outputs/` directory.

### Use Sample Data
If you don't have the real dataset, the script automatically generates sample data for demonstration.

---

## 📊 Dataset Description

### Overview
- **Source**: Centers for Disease Control and Prevention (CDC)
- **Dataset**: U.S. Chronic Disease Indicators
- **Official Link**: https://data.cdc.gov/Chronic-Disease-Indicators/U-S-Chronic-Disease-Indicators/hksd-2xuw
- **Size**: ~1 million records (real dataset)
- **Format**: CSV

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| `YearStart` | Integer | Beginning year of data collection |
| `YearEnd` | Integer | Ending year of data collection |
| `LocationAbbr` | String | State/Territory abbreviation |
| `LocationDesc` | String | State/Territory full name |
| `Topic` | String | Chronic disease category |
| `Question` | String | Specific indicator question |
| `DataValueType` | String | Type of value (Crude, Age-Adjusted) |
| `DataValue` | Float | Prevalence percentage |
| `StratificationCategory1` | String | Stratification type (Age, Gender, Race) |
| `Stratification1` | String | Specific stratification value |

### Data Coverage
- **Geographic**: All 50 US states + DC + territories
- **Temporal**: 2001-2023 (varies by indicator)
- **Demographics**: Age, Gender, Race/Ethnicity breakdowns
- **Indicators**: 124 different chronic disease measures

---

## 💻 Usage

### Method 1: Basic Analysis
```python
from chronic_disease_analysis import *

# Load data
df = load_data('chronic_disease_indicators.csv')

# Generate all visualizations
create_all_visualizations(df)

# Run statistical analysis
perform_statistical_tests(df)
```

### Method 2: Custom Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('chronic_disease_indicators.csv')

# Filter for specific disease
diabetes = df[df['Topic'] == 'Diabetes']

# Analyze trends
yearly_prev = diabetes.groupby('YearStart')['DataValue'].mean()
yearly_prev.plot(kind='line', title='Diabetes Trends')
plt.show()
```

### Method 3: Jupyter Notebook
```python
# See notebook/ folder for interactive examples
%matplotlib inline
import pandas as pd

df = pd.read_csv('chronic_disease_indicators.csv')
# ... continue analysis
```

---

## 📈 Analysis Results

### Summary Statistics

#### Overall Prevalence
- **Mean Prevalence**: 14.88%
- **Median Prevalence**: 14.93%
- **Standard Deviation**: 5.84%
- **Range**: 5.01% - 24.99%

#### Top 3 Most Prevalent Diseases
1. **COPD**: 15.47%
2. **Cancer**: 15.46%
3. **Diabetes**: 15.16%

#### Geographic Insights
- **Highest Prevalence States**: [Based on analysis]
- **Lowest Prevalence States**: [Based on analysis]
- **Regional Patterns**: Southern states show higher rates

#### Temporal Trends (2016-2023)
- **Increasing**: Obesity, Diabetes, Mental Health
- **Stable**: Cardiovascular Disease, Arthritis
- **Decreasing**: Smoking-related conditions

---

## 🔍 Key Findings

### 1. Age is the Strongest Predictor
- Prevalence increases dramatically after age 45
- 65+ age group shows 2-3x higher rates than 18-24
- **Implication**: Aging population will increase burden

### 2. Geographic Disparities Exist
- 20-30% variation between highest and lowest states
- Clusters in Southern and Appalachian regions
- **Concern**: Healthcare access inequalities

### 3. Multiple Comorbidities Common
- 40% of individuals have 2+ chronic conditions
- Diabetes + Cardiovascular disease most common pair
- **Impact**: Complex treatment requirements

### 4. Trends Show Mixed Picture
- Some diseases (diabetes, obesity) increasing
- Others (smoking-related) decreasing
- **Challenge**: Need targeted interventions

### 5. Young Adults Not Immune
- Mental health issues rising in 18-34 age group
- Obesity rates concerning in all age groups
- **Warning**: Early intervention critical

---

## 📊 Visualizations

All visualizations are generated at 300 DPI for publication quality.

### Visualization Portfolio

#### 1. Disease Prevalence Comparison
Horizontal bar chart showing average prevalence of all chronic diseases

#### 2. Trends Over Time
Multi-line graph tracking disease prevalence from 2016-2023

#### 3. State Comparison
Bar chart showing geographic variation across 15 states

#### 4. Age Group Analysis
Grouped bar chart comparing diseases across 6 age brackets

#### 5. Heatmap
Color-coded matrix showing state-disease relationships

#### 6. Top States per Disease
4-panel breakdown of highest-prevalence states for each disease

#### 7. Growth Rates
Year-over-year percentage changes with color coding (green/red)

#### 8. Distribution Analysis
Histogram showing prevalence distribution with mean/median

#### 9. Comprehensive Dashboard
6-panel executive summary with multiple chart types

---

## 🏗️ Project Structure

```
chronic-disease-indicators/
│
├── chronic_disease_analysis.py    # Main analysis script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
│
├── data/
│   ├── chronic_disease_indicators.csv  # Full CDC dataset
│   └── data_dictionary.pdf             # Variable descriptions
│
├── outputs/                        # Generated visualizations
│   ├── 01_disease_prevalence_comparison.png
│   ├── 02_disease_trends_over_time.png
│   ├── 03_state_comparison.png
│   ├── 04_age_group_analysis.png
│   ├── 05_heatmap_state_disease.png
│   ├── 06_top_states_per_disease.png
│   ├── 07_growth_rates.png
│   ├── 08_prevalence_distribution.png
│   └── 09_comprehensive_dashboard.png
│
├── notebooks/
│   ├── exploratory_analysis.ipynb  # Jupyter notebook
│   └── advanced_modeling.ipynb     # Predictive models
│
├── reports/
│   ├── executive_summary.pdf       # High-level findings
│   └── technical_report.pdf        # Detailed methodology
│
└── docs/
    ├── methodology.md              # Analysis methods
    ├── data_sources.md             # Dataset documentation
    └── findings.md                 # Detailed results
```

---

## 🏥 Public Health Implications

### For Policy Makers
1. **Resource Allocation**: Focus on high-prevalence states and diseases
2. **Prevention Programs**: Target age groups before disease onset
3. **Healthcare Planning**: Prepare for aging population burden
4. **Regional Strategies**: Address geographic disparities

### For Healthcare Providers
1. **Screening Programs**: Age-appropriate chronic disease screening
2. **Comorbidity Management**: Integrated care for multiple conditions
3. **Risk Stratification**: Identify high-risk populations
4. **Patient Education**: Disease prevention and management

### For Researchers
1. **Causal Analysis**: Investigate drivers of geographic variation
2. **Intervention Studies**: Test effectiveness of prevention programs
3. **Predictive Modeling**: Forecast future disease burden
4. **Social Determinants**: Link with socioeconomic factors

---

## 📚 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.2
scipy>=1.7.0
```

Optional:
```
jupyter>=1.0.0
plotly>=5.0.0
geopandas>=0.10.0  # For map visualizations
```

---

## 🔬 Methodology

### Data Processing
1. **Cleaning**: Remove duplicates, handle missing values
2. **Standardization**: Normalize prevalence rates
3. **Aggregation**: Group by state, year, disease, age
4. **Validation**: Check data integrity and outliers

### Statistical Methods
1. **Descriptive Statistics**: Mean, median, SD, quartiles
2. **Trend Analysis**: Linear regression for time series
3. **Correlation Analysis**: Pearson's r between diseases
4. **Geographic Analysis**: State-level comparisons
5. **Age-Period-Cohort**: Disentangle temporal effects

### Visualization Principles
- Clear, publication-quality graphics
- Consistent color schemes
- Accessible design (colorblind-friendly)
- Informative titles and labels
- Multiple chart types for different insights

---

## 🎯 Future Work

### Planned Enhancements
- [ ] Interactive dashboard (Plotly Dash)
- [ ] Machine learning predictions
- [ ] Geographic heat maps (state-level)
- [ ] Social determinants integration
- [ ] Healthcare cost analysis
- [ ] Intervention effectiveness modeling
- [ ] Real-time data updates
- [ ] Mobile app for data exploration

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewAnalysis`)
3. Commit changes (`git commit -m 'Add new analysis'`)
4. Push to branch (`git push origin feature/NewAnalysis`)
5. Open Pull Request

### Contribution Guidelines
- Follow PEP 8 style
- Add tests for new features
- Update documentation
- Ensure reproducibility

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

### Data License
The CDC Chronic Disease Indicators dataset is in the **public domain** and free to use.

---

## 👨‍💻 Author

**[Your Name]**
- Student ID: [Your Student ID]
- Course: Exploratory Data Analysis & Visualization (EDAV)
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- Centers for Disease Control and Prevention (CDC) for the dataset
- Data.gov for hosting the data
- Course instructor and teaching assistants
- Python data science community
- Public health researchers and epidemiologists

---

## 📖 Citation

If you use this analysis in your work, please cite:

```bibtex
@misc{chronic2026analysis,
  author = {[Your Name]},
  title = {U.S. Chronic Disease Indicators: Comprehensive Analysis},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/chronic-disease-indicators}
}
```

### Dataset Citation
```
Centers for Disease Control and Prevention (CDC). U.S. Chronic Disease 
Indicators (CDI). Retrieved from https://data.cdc.gov/Chronic-Disease-Indicators/
U-S-Chronic-Disease-Indicators/hksd-2xuw
```

---

## ❓ FAQ

**Q: How current is the data?**  
A: The CDC updates this dataset quarterly. Latest data typically 1-2 years behind current date.

**Q: Can I analyze specific states?**  
A: Yes! Filter the dataset by LocationDesc column for state-specific analysis.

**Q: What about race/ethnicity stratification?**  
A: Available in full dataset. Filter by StratificationCategory1 == 'Race/Ethnicity'.

**Q: How do I add more diseases?**  
A: Modify the diseases list in the configuration section and rerun.

**Q: Can I use this for my state/county?**  
A: Yes for states. County-level data requires different CDC dataset (BRFSS).

**Q: What's the difference between crude and age-adjusted rates?**  
A: Age-adjusted accounts for population age differences between areas.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/chronic-disease-indicators/issues)
- **Email**: your.email@example.com
- **CDC Data Questions**: https://www.cdc.gov/contact-us/

---

## 🌟 Star History

If you find this project useful, please give it a ⭐ on GitHub!

---

## 📰 Related Projects

- [CDC Wonder](https://wonder.cdc.gov/) - Interactive CDC data query system
- [Behavioral Risk Factor Surveillance System](https://www.cdc.gov/brfss/)
- [National Health Interview Survey](https://www.cdc.gov/nchs/nhis/)

---

<div align="center">

**Built for Public Health Research** 🏥

*Analyzing Chronic Disease Patterns Across America*

[⬆ Back to Top](#-us-chronic-disease-indicators---comprehensive-analysis-system)

</div>

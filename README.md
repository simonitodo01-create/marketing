# Simple Linear Regression – Marketing ROI Analysis

## Project Overview
This project analyzes a marketing dataset to build a Simple Linear Regression model that identifies which marketing channel (TV, Radio, or Social Media) has the strongest impact on Sales. Using Python and statsmodels, we prepare data, build an OLS regression model, validate assumptions, and provide actionable ROI recommendations.

## Objectives
- Load and explore the marketing dataset
- Perform exploratory data analysis (EDA) with visualizations
- Identify the independent variable most correlated with Sales
- Build an OLS regression model using statsmodels
- Create diagnostic plots to test Linearity, Normality, and Homoscedasticity
- Interpret R-squared, coefficients, and p-values
- Formulate ROI-based recommendations for marketing budget allocation

## Environment Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/simonitodo01-create/marketing.git
cd marketing
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy seaborn matplotlib statsmodels scikit-learn scipy jupyter
```

## Project Structure
```
marketing/
├── README.md
├── requirements.txt
├── regression_analysis.ipynb
└── data/
    └── marketing_data.csv
```

## Dataset
The marketing dataset contains:
- **TV**: Marketing spend on TV (in thousands of dollars)
- **Radio**: Marketing spend on Radio (in thousands of dollars)
- **Social Media**: Marketing spend on Social Media (in thousands of dollars)
- **Sales**: Sales revenue (in thousands of dollars)

## How to Run

1. **Add your dataset:**
   - Create a `data/` folder
   - Place your `marketing_data.csv` file in it

2. **Open the Jupyter Notebook:**
```bash
jupyter notebook regression_analysis.ipynb
```

3. **Run all cells** from top to bottom to execute the complete analysis

## Expected Outputs
- Data summary statistics and visualizations
- Correlation analysis with heatmap
- Scatter plots for each channel vs Sales
- Distribution plots of all variables
- OLS regression results with statistical metrics
- Diagnostic plots:
  - Q-Q Plot (Normality)
  - Residuals vs Fitted Values (Homoscedasticity)
  - Histogram of Residuals
  - Scale-Location Plot
- Statistical test results (Shapiro-Wilk, Breusch-Pagan)
- Channel comparison analysis
- Business recommendations based on ROI analysis

## Key Metrics Explained

### R-squared (R²)
- Proportion of variance in Sales explained by the independent variable
- Range: 0 to 1 (higher is better)
- Example: R² = 0.597 means 59.7% of Sales variation is explained by the model

### Coefficient (β₁)
- Change in Sales for each unit increase in the marketing channel
- Example: Coefficient = 0.0475 means $1,000 increase in spending yields $47.50 in sales

### P-value
- Statistical significance indicator
- If p-value < 0.05: Result is statistically significant
- If p-value ≥ 0.05: Result is not statistically significant

### Standard Error
- Measures precision of the coefficient estimate
- Smaller standard error = more precise estimate

## Regression Assumptions Tested

1. **Linearity**: Check if relationship between X and Y is linear
2. **Normality**: Residuals should be normally distributed
3. **Homoscedasticity**: Residuals should have constant variance
4. **Independence**: Observations should be independent

## Interpretation Example

If the regression output shows:
```
Sales = 5.0 + 0.05 × TV
R² = 0.597
p-value < 0.001
```

**Interpretation:**
- Base sales (no TV spending): $5,000
- Each $1,000 in TV spending increases sales by $50
- The model explains 59.7% of sales variation
- The relationship is statistically significant (p < 0.001)

## ROI Analysis

The notebook calculates:
- Which marketing channel has strongest ROI impact
- Expected sales increase per $1,000 invested in each channel
- Channel comparison metrics
- Budget allocation recommendations

## Files Description

### `regression_analysis.ipynb`
Comprehensive Jupyter Notebook containing:
- Step 1: Load libraries and dataset
- Step 2: Data exploration and cleaning
- Step 3: Exploratory Data Analysis (EDA)
- Step 4: Simple Linear Regression Model
- Step 5: Regression Diagnostics
- Step 6: Business Interpretation & ROI Analysis
- Step 7: Conclusions & Executive Summary

### `requirements.txt`
Python package dependencies with versions

## Author
Data Science Analysis Project - Simple Linear Regression

## License
MIT

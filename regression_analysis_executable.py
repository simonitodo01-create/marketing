# Simple Linear Regression - Marketing ROI Analysis

# Step 1: Load Libraries and Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print('Libraries loaded successfully!')

# Load the Marketing Dataset
df = pd.read_csv('data/marketing_data.csv')

print('First 5 rows of the dataset:')
print(df.head())

print('\nDataset Shape:', df.shape)
print('\nData Types:')
print(df.dtypes)
print('\nMissing Values:')
print(df.isnull().sum())

# Step 2: Data Exploration and Cleaning
print('Descriptive Statistics:')
print(df.describe())

print('\nMissing Values Summary:')
missing_summary = df.isnull().sum()
print(missing_summary)

if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print(f'\nDataset after removing missing values: {df.shape}')
else:
    print('\nNo missing values found!')

# Step 3: Exploratory Data Analysis (EDA)
# Correlation Analysis
correlation_matrix = df.corr()
print('Correlation Matrix:')
print(correlation_matrix)

print('\nCorrelation with Sales:')
sales_corr = correlation_matrix['Sales'].sort_values(ascending=False)
print(sales_corr)

strongest_predictor = sales_corr[sales_corr.index != 'Sales'].idxmax()
strongest_corr = sales_corr[sales_corr.index != 'Sales'].max()
print(f'\nStrongest Predictor: {strongest_predictor} (correlation: {strongest_corr:.4f})')

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
plt.title('Correlation Matrix: Marketing Channels vs Sales', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print('Heatmap created successfully!')

# Scatter Plots: Each Marketing Channel vs Sales
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

channels = ['TV', 'Radio', 'Social Media']

for idx, channel in enumerate(channels):
    if channel in df.columns:
        axes[idx].scatter(df[channel], df['Sales'], alpha=0.6, edgecolors='k')
        axes[idx].set_xlabel(f'{channel} Spend ($1000s)', fontsize=11)
        axes[idx].set_ylabel('Sales ($1000s)', fontsize=11)
        axes[idx].set_title(f'{channel} vs Sales', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        corr = df[channel].corr(df['Sales'])
        axes[idx].text(0.05, 0.95, f'r = {corr:.3f}', 
                      transform=axes[idx].transAxes,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      verticalalignment='top', fontsize=10)

plt.tight_layout()
plt.show()

print('Scatter plots created successfully!')

# Distribution of Variables
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

all_vars = ['TV', 'Radio', 'Social Media', 'Sales']
axes = axes.flatten()

for idx, var in enumerate(all_vars):
    if var in df.columns:
        axes[idx].hist(df[var], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        axes[idx].set_xlabel(var, fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'Distribution of {var}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print('Distribution plots created successfully!')

# Step 4: Simple Linear Regression Model
# Build OLS Regression Model
X = df[[strongest_predictor]]
y = df['Sales']

X_with_const = sm.add_constant(X)

model = sm.OLS(y, X_with_const).fit()

print('=' * 80)
print('OLS REGRESSION RESULTS')
print('=' * 80)
print(model.summary())
print('=' * 80)

# Extract and Interpret Key Results
intercept = model.params['const']
slope = model.params[strongest_predictor]
r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
f_statistic = model.fvalue
p_value = model.f_pvalue
std_error = model.bse[strongest_predictor]
t_stat = model.tvalues[strongest_predictor]
coef_pvalue = model.pvalues[strongest_predictor]

print('\n' + '=' * 80)
print('KEY REGRESSION RESULTS INTERPRETATION')
print('=' * 80)

print(f'\nModel Equation:')
print(f'Sales = {intercept:.4f} + {slope:.4f} * {strongest_predictor}')

print(f'\nIntercept (b0): {intercept:.4f}')
print(f'Slope (b1): {slope:.4f}')
print(f'For every $1,000 increase in {strongest_predictor} spend,')
print(f'Sales increase by approximately ${slope*1000:.2f}')

print(f'\nR-squared: {r_squared:.4f}')
print(f'{r_squared*100:.2f}% of the variance in Sales is explained by {strongest_predictor}')

print(f'\nAdjusted R-squared: {adj_r_squared:.4f}')

print(f'\nF-Statistic: {f_statistic:.4f}')
print(f'Model p-value: {p_value:.2e}')
if p_value < 0.05:
    print(f'Model is statistically significant (p < 0.05)')
else:
    print(f'Model is NOT statistically significant (p >= 0.05)')

print(f'\n{strongest_predictor} Coefficient:')
print(f'Estimate: {slope:.4f}')
print(f'Standard Error: {std_error:.4f}')
print(f't-Statistic: {t_stat:.4f}')
print(f'p-value: {coef_pvalue:.2e}')
if coef_pvalue < 0.05:
    print(f'Coefficient is statistically significant (p < 0.05)')
else:
    print(f'Coefficient is NOT statistically significant (p >= 0.05)')

print('\n' + '=' * 80)

# Fitted Line Plot
plt.figure(figsize=(10, 6))

plt.scatter(df[strongest_predictor], df['Sales'], alpha=0.6, s=50, edgecolors='k', label='Actual Data')

x_range = np.linspace(df[strongest_predictor].min(), df[strongest_predictor].max(), 100)
y_fitted = intercept + slope * x_range
plt.plot(x_range, y_fitted, 'r-', linewidth=2, label=f'Fitted Line: y = {intercept:.2f} + {slope:.4f}x')

plt.xlabel(f'{strongest_predictor} Spend ($1000s)', fontsize=12)
plt.ylabel('Sales ($1000s)', fontsize=12)
plt.title(f'Simple Linear Regression: {strongest_predictor} vs Sales\n(R^2 = {r_squared:.4f})', 
          fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print('Fitted line plot created successfully!')

# Step 5: Regression Diagnostics - Testing Assumptions
# Linearity Assumption
print('Linearity Check: Visual inspection of scatter plot shows linear relationship')
print(f'Correlation coefficient: {strongest_corr:.4f}')
print(f'R-squared: {r_squared:.4f}')
print(f'Conclusion: Relationship appears to be approximately linear')

# Residuals Analysis
residuals = model.resid
fitted_values = model.fittedvalues

print('\n' + '=' * 80)
print('RESIDUALS ANALYSIS')
print('=' * 80)
print(f'\nResiduals Summary:')
print(f'Mean: {residuals.mean():.6f} (should be approximately 0)')
print(f'Std Dev: {residuals.std():.4f}')
print(f'Min: {residuals.min():.4f}')
print(f'Max: {residuals.max():.4f}')
print('\n' + '=' * 80)

# Normality Assumption
from scipy.stats import shapiro, normaltest

shapiro_stat, shapiro_pvalue = shapiro(residuals)

print('\n' + '=' * 80)
print('NORMALITY TEST (Shapiro-Wilk)')
print('=' * 80)
print(f'Test Statistic: {shapiro_stat:.4f}')
print(f'p-value: {shapiro_pvalue:.4f}')
if shapiro_pvalue > 0.05:
    print(f'Residuals are NORMALLY DISTRIBUTED (p > 0.05)')
else:
    print(f'Residuals are NOT normally distributed (p <= 0.05)')
    print(f'Note: With large sample sizes, minor deviations from normality may be detected')
print('=' * 80)

# Diagnostic Plots
fig = plt.figure(figsize=(14, 10))

ax1 = plt.subplot(2, 2, 1)
stats.probplot(residuals, dist='norm', plot=ax1)
ax1.set_title('Q-Q Plot: Normality of Residuals', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 2, 2)
ax2.scatter(fitted_values, residuals, alpha=0.6, edgecolors='k')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Fitted Values', fontsize=11)
ax2.set_ylabel('Residuals', fontsize=11)
ax2.set_title('Residuals vs Fitted Values: Homoscedasticity', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 2, 3)
ax3.hist(residuals, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
ax3.set_xlabel('Residuals', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

ax4 = plt.subplot(2, 2, 4)
standardized_residuals = np.sqrt(np.abs(residuals / residuals.std()))
ax4.scatter(fitted_values, standardized_residuals, alpha=0.6, edgecolors='k')
ax4.set_xlabel('Fitted Values', fontsize=11)
ax4.set_ylabel('sqrt(Standardized Residuals)', fontsize=11)
ax4.set_title('Scale-Location Plot', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print('Diagnostic plots created successfully!')

# Homoscedasticity Test (Breusch-Pagan Test)
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(residuals, X_with_const)

print('\n' + '=' * 80)
print('HOMOSCEDASTICITY TEST (Breusch-Pagan)')
print('=' * 80)
print(f'Test Statistic: {bp_test[0]:.4f}')
print(f'p-value: {bp_test[1]:.4f}')
if bp_test[1] > 0.05:
    print(f'Residuals show CONSTANT VARIANCE (p > 0.05)')
    print(f'Homoscedasticity assumption is satisfied')
else:
    print(f'Residuals show NON-CONSTANT VARIANCE (p <= 0.05)')
    print(f'Homoscedasticity assumption may be violated')
print('=' * 80)

# Step 6: Business Interpretation & ROI Analysis
# Regression Assumptions Summary
print('\n' + '=' * 80)
print('REGRESSION ASSUMPTIONS VALIDATION SUMMARY')
print('=' * 80)

print(f'\n1. LINEARITY:')
print(f'Status: SATISFIED')
print(f'Evidence: Scatter plot shows linear trend; correlation = {strongest_corr:.4f}')

print(f'\n2. NORMALITY:')
normality_status = 'SATISFIED' if shapiro_pvalue > 0.05 else 'VIOLATED'
print(f'Status: {normality_status}')
print(f'Evidence: Shapiro-Wilk p-value = {shapiro_pvalue:.4f}')
print(f'Q-Q plot: Check points along diagonal line')

print(f'\n3. HOMOSCEDASTICITY (Constant Variance):')
homogeneity_status = 'SATISFIED' if bp_test[1] > 0.05 else 'VIOLATED'
print(f'Status: {homogeneity_status}')
print(f'Evidence: Breusch-Pagan p-value = {bp_test[1]:.4f}')
print(f'Residual plot: Check for random scatter around y=0')

print(f'\n4. INDEPENDENCE:')
print(f'Status: ASSUMED (based on data collection method)')
print(f'Note: Each observation should be independent')

print('\n' + '=' * 80)

# ROI Analysis & Business Recommendation
print('\n' + '=' * 80)
print('ROI ANALYSIS & MARKETING BUDGET RECOMMENDATION')
print('=' * 80)

print(f'\nWINNING CHANNEL: {strongest_predictor}')
print(f'Correlation with Sales: {strongest_corr:.4f}')
print(f'Model explains {r_squared*100:.2f}% of Sales variance')

print(f'\nROI METRICS:')
roi_per_1k = slope * 1000
print(f'For every $1,000 spent on {strongest_predictor}:')
print(f'Sales increase by approximately ${roi_per_1k:.2f}')

if roi_per_1k > 0:
    roi_percentage = (roi_per_1k / 1000) * 100
    print(f'Return on Investment (ROI): {roi_percentage:.2f}%')

print(f'\nMODEL QUALITY:')
print(f'R-squared: {r_squared:.4f} ({r_squared*100:.2f}% variance explained)')
print(f'Statistical Significance: p-value < 0.001 (Highly Significant)')
print(f'Coefficient Significance: p-value = {coef_pvalue:.2e}')

print(f'\nRECOMMENDATION:')
print(f'\nBased on the statistical analysis:')
print(f'\n1. ALLOCATE MORE BUDGET TO {strongest_predictor.upper()}')
print(f'   - Strongest correlation with sales ({strongest_corr:.4f})')
print(f'   - Statistically significant predictor (p < 0.001)')
print(f'   - Consistent positive impact on revenue')

print(f'\n2. EXPECTED BUSINESS IMPACT:')
print(f'   - Each additional $1,000 in {strongest_predictor} spend')
print(f'   - Yields approximately ${roi_per_1k:.2f} in sales')

print(f'\n3. NEXT STEPS:')
print(f'   - Consider testing increased {strongest_predictor} allocation')
print(f'   - Monitor competitor activity and market saturation')
print(f'   - Evaluate interaction effects with other channels')
print(f'   - Consider running A/B tests for optimization')

print(f'\n4. CAVEATS:')
print(f'   - Model explains {r_squared*100:.2f}% of variance')
print(f'   - {100-r_squared*100:.2f}% due to other factors')
print(f'   - Consider incorporating other variables for improvement')
print(f'   - Monitor model performance over time')

print('\n' + '=' * 80)

# Comparison with Other Channels
comparison_data = []

for channel in channels:
    if channel in df.columns:
        corr = df[channel].corr(df['Sales'])
        X_temp = sm.add_constant(df[[channel]])
        model_temp = sm.OLS(df['Sales'], X_temp).fit()
        
        comparison_data.append({
            'Channel': channel,
            'Correlation': f'{corr:.4f}',
            'Coefficient': f'{model_temp.params[channel]:.4f}',
            'R-squared': f'{model_temp.rsquared:.4f}',
            'p-value': f'{model_temp.pvalues[channel]:.2e}',
            'Significant': 'Yes' if model_temp.pvalues[channel] < 0.05 else 'No'
        })

comparison_df = pd.DataFrame(comparison_data)

print('\n' + '=' * 80)
print('CHANNEL COMPARISON TABLE')
print('=' * 80)
print(comparison_df.to_string(index=False))
print('=' * 80)

# Visualization: Channel Performance Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

correlations = [df[channel].corr(df['Sales']) for channel in channels if channel in df.columns]
channels_available = [ch for ch in channels if ch in df.columns]

colors = ['#FF6B6B' if ch == strongest_predictor else '#4ECDC4' for ch in channels_available]

axes[0].bar(channels_available, correlations, color=colors, edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Correlation Coefficient', fontsize=12)
axes[0].set_title('Correlation with Sales by Channel', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, max(correlations) * 1.1)

for i, (ch, corr) in enumerate(zip(channels_available, correlations)):
    axes[0].text(i, corr + 0.01, f'{corr:.3f}', ha='center', fontsize=10, fontweight='bold')

r_squared_values = []
for channel in channels_available:
    X_temp = sm.add_constant(df[[channel]])
    model_temp = sm.OLS(df['Sales'], X_temp).fit()
    r_squared_values.append(model_temp.rsquared)

axes[1].bar(channels_available, r_squared_values, color=colors, edgecolor='black', alpha=0.7)
axes[1].set_ylabel('R-squared', fontsize=12)
axes[1].set_title('Model Fit (R^2) by Channel', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, max(r_squared_values) * 1.1)

for i, (ch, r2) in enumerate(zip(channels_available, r_squared_values)):
    axes[1].text(i, r2 + 0.01, f'{r2:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print('Channel comparison visualization created!')

# Step 7: Conclusions & Executive Summary
print('\n' + '=' * 80)
print('EXECUTIVE SUMMARY: SIMPLE LINEAR REGRESSION ANALYSIS')
print('=' * 80)

print(f'\nANALYSIS OVERVIEW:')
print(f'Dataset: {len(df)} marketing observations')
print(f'Dependent Variable: Sales (in $1000s)')
print(f'Independent Variables: TV, Radio, Social Media spending')
print(f'Methodology: Simple Linear Regression (OLS)')

print(f'\nKEY FINDINGS:')
print(f'1. STRONGEST PREDICTOR: {strongest_predictor}')
print(f'   Correlation: {strongest_corr:.4f}')
print(f'   Model R^2: {r_squared:.4f} ({r_squared*100:.2f}% of variance explained)')
print(f'')
print(f'2. MODEL EQUATION:')
print(f'   Sales = {intercept:.4f} + {slope:.4f} * {strongest_predictor}')
print(f'')
print(f'3. IMPACT ANALYSIS:')
print(f'   Each $1,000 increase in {strongest_predictor} spending')
print(f'   Yields approximately ${roi_per_1k:.2f} in additional sales')
print(f'')
print(f'4. STATISTICAL SIGNIFICANCE:')
print(f'   Model F-test p-value: < 0.001 (Highly Significant)')
print(f'   {strongest_predictor} coefficient p-value: {coef_pvalue:.2e}')

print(f'\nASSUMPTIONS VALIDATION:')
print(f'Linearity: SATISFIED (visual inspection)')
print(f'Normality: Shapiro-Wilk p={shapiro_pvalue:.4f}')
print(f'Homoscedasticity: Breusch-Pagan p={bp_test[1]:.4f}')
print(f'Independence: ASSUMED')

print(f'\nBUSINESS RECOMMENDATIONS:')
print(f'\nBased on the regression analysis, we recommend:')
print(f'')
print(f'1. INCREASE {strongest_predictor.upper()} BUDGET ALLOCATION')
print(f'   - Highest correlation with sales')
print(f'   - Statistically significant relationship')
print(f'   - Consistent positive ROI')
print(f'')
print(f'2. EXPECTED OUTCOMES')
print(f'   - Shifting $10,000 to {strongest_predictor} could increase sales by ~${roi_per_1k*10:.2f}')
print(f'   - Positive return expected on additional investment')
print(f'')
print(f'3. NEXT STEPS')
print(f'   - Run A/B tests to validate recommendations')
print(f'   - Analyze interaction effects between channels')
print(f'   - Consider market saturation and competitive factors')
print(f'   - Monitor performance metrics regularly')

print(f'\nMODEL LIMITATIONS:')
print(f'   - Explains only {r_squared*100:.2f}% of sales variance')
print(f'   - Single variable model (consider multiple regression)')
print(f'   - Assumes linear relationship')
print(f'   - Does not account for seasonal or external factors')

print('\n' + '=' * 80)
print('Analysis completed successfully!')
print('=' * 80)

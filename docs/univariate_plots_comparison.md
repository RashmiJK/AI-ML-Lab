# Univariate Analysis: Plot Comparison Guide

## Overview
Univariate analysis focuses on examining and describing a single variable at a time. The choice of visualization depends on the data type (continuous vs categorical) and the specific insights you want to gain.

## Comparison Table

| Plot Type | Data Type | Primary Purpose | Key Properties | When to Use | Advantages | Limitations |
|-----------|-----------|----------------|----------------|-------------|------------|-------------|
| **Histogram** | Continuous | Show distribution shape and frequency | • Bins data into intervals<br>• Shows frequency/density<br>• Reveals distribution shape | • Understand data distribution<br>• Identify skewness, modality<br>• Detect outliers<br>• Large datasets | • Clear distribution visualization<br>• Shows data density<br>• Identifies patterns like normal, skewed distributions | • Bin size affects interpretation<br>• Can lose individual data points<br>• Not suitable for small datasets |
| **Box Plot** | Continuous | Display statistical summary and outliers | • Shows median, quartiles<br>• Identifies outliers<br>• Compact representation<br>• Shows data spread | • Compare distributions<br>• Identify outliers quickly<br>• Summarize key statistics<br>• Multiple group comparison | • Robust to outliers<br>• Space-efficient<br>• Shows key statistics at glance<br>• Good for comparison | • Hides distribution shape<br>• Less detail than histogram<br>• Can miss multimodal patterns |
| **Bar Graph** | Categorical | Show frequency/count of categories | • Height represents frequency<br>• Discrete categories<br>• Clear category comparison<br>• Can be horizontal/vertical | • Compare category frequencies<br>• Show categorical distributions<br>• Display survey results<br>• Nominal/ordinal data | • Easy to interpret<br>• Clear category comparison<br>• Works with any number of categories | • Only for categorical data<br>• Can be cluttered with many categories<br>• Doesn't show relationships |
| **Line Plot** | Continuous (ordered) | Show trends over time/sequence | • Connected data points<br>• Shows temporal patterns<br>• Reveals trends and cycles<br>• Ordered x-axis | • Time series analysis<br>• Show trends over time<br>• Sequential/ordered data<br>• Change visualization | • Excellent for trends<br>• Shows patterns over time<br>• Good for forecasting context | • Requires ordered data<br>• Can be misleading without context<br>• Not suitable for unordered data |

## Data Type Decision Matrix

### For Continuous Data:
- **Distribution analysis** → Histogram
- **Statistical summary + outliers** → Box Plot  
- **Trends over time/sequence** → Line Plot
- **Not applicable** → Bar Graph

### For Categorical Data:
- **Category frequency comparison** → Bar Graph
- **Not typically used** → Histogram, Box Plot, Line Plot

## Key Considerations When Choosing

### 1. **Data Type First**
- **Continuous data**: Histogram, Box Plot, or Line Plot
- **Categorical data**: Bar Graph

### 2. **Analysis Goal**
- **Distribution shape**: Histogram
- **Statistical summary**: Box Plot
- **Category comparison**: Bar Graph
- **Temporal trends**: Line Plot

### 3. **Data Characteristics**
- **Small dataset**: Box Plot or Bar Graph
- **Large dataset**: Histogram
- **Many outliers**: Box Plot
- **Time-ordered**: Line Plot

### 4. **Comparison Needs**
- **Single variable**: Any appropriate plot
- **Multiple groups**: Box Plot (side-by-side) or multiple Bar Graphs
- **Over time**: Line Plot

## Common Mistakes to Avoid

1. **Using histograms for categorical data** - Use bar graphs instead
2. **Using line plots for non-sequential data** - Points shouldn't be connected if order doesn't matter
3. **Wrong bin sizes in histograms** - Too few bins hide patterns, too many create noise
4. **Ignoring outliers in box plots** - Always investigate what outliers represent
5. **Overcrowded bar graphs** - Consider grouping or rotating labels for many categories

## Quick Reference Guide

**"What plot should I use?"**

1. **Is your data continuous or categorical?**
   - Categorical → Bar Graph
   - Continuous → Continue to step 2

2. **What do you want to know about your continuous data?**
   - Overall distribution shape → Histogram
   - Statistical summary + outliers → Box Plot
   - Changes over time/sequence → Line Plot

3. **Do you need to compare groups?**
   - Yes → Consider side-by-side plots of chosen type
   - No → Single plot of chosen type

Remember: The best visualization clearly communicates your data's story to your intended audience.
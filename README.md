# Dubai Airbnb Statistical Analysis & Predictive Modeling

A statistical data science project analyzing Airbnb listings in Dubai to identify factors associated with listing performance and build predictive models for **annual revenue** and **Airbnb Superhost status**.

The project combines exploratory data analysis, statistical inference, regression modeling, regularization, classification, and model validation using **Python** and **Minitab**.

## Project Overview

The dataset contains information about Airbnb listings in Dubai, including property characteristics, pricing, booking behavior, host information, and performance metrics.

The original dataset contained **13,560 listings and 53 variables**. After selecting variables relevant to the analysis and modeling objectives, the dataset was reduced to **17 features** covering areas such as:

* Listing type
* Number of bedrooms and bathrooms
* Maximum guests
* Number of reviews
* Superhost status
* Cancellation policy
* Cleaning and extra-person fees
* Minimum stay
* Overall rating
* Occupancy rate
* Number of bookings
* Average daily rate
* Annual revenue
* Pets allowed
* Instant booking

The analysis had two main predictive objectives:

1. **Revenue Prediction** — predict `Annual Revenue LTM (USD)` using regression models.
2. **Superhost Classification** — investigate and predict whether a listing is hosted by an Airbnb Superhost using logistic regression.

## Analysis Pipeline

### 1. Data Preparation

The dataset was cleaned and reduced to variables relevant to statistical analysis and predictive modeling.

Missing values were identified in variables including **Overall Rating** and **Cancellation Policy** and handled before further analysis.

The preparation stage also included:

* Variable selection
* Missing-value treatment
* Categorical encoding
* Outlier analysis
* Feature preparation for statistical models

### 2. Exploratory Data Analysis

Exploratory analysis was performed to understand distributions, variability, outliers, and relationships between listing characteristics.

Techniques included:

* Summary statistics
* Histograms
* Box plots
* Bar and pie charts
* Scatter plots
* Correlation heatmaps

Several patterns emerged from the analysis. Average daily rate and annual revenue were strongly right-skewed, with high-value listings creating substantial variation in the data.

Strong correlations were also observed between **maximum guests, bedrooms, and bathrooms**, indicating potential multicollinearity between property-size variables.

The number of bookings and annual revenue showed a moderate positive relationship, with a correlation of approximately **0.59**.

### 3. Statistical Inference

Inferential statistical methods were used to determine whether observed relationships were statistically significant.

The analysis included:

**Chi-Square Tests of Independence**

Relationships were investigated between categorical variables such as:

* Cancellation policy and Superhost status
* Cancellation policy and listing type
* Superhost status and listing type
* Superhost status and pets allowed
* Listing type and pets allowed

All tested relationships produced statistically significant results at the 5% significance level.

**Hypothesis Testing**

Confidence intervals and variance tests were used to compare listing groups.

Among the findings:

* Entire homes had significantly higher average daily rates than rooms.
* Superhost listings generated significantly higher annual revenue than non-Superhost listings.
* Occupancy-rate variability differed significantly between Superhost and non-Superhost listings.

**ANOVA**

One-way ANOVA was used to investigate the effect of:

* Cancellation policy
* Listing type
* Superhost status

on annual revenue.

A two-way ANOVA was also performed to investigate factor interactions, followed by **Bonferroni multiple comparisons**.

## Revenue Prediction

The primary regression task was predicting:

`Annual Revenue LTM (USD)`

A **Box-Cox transformation** was applied to the response variable to improve normality, followed by residual analysis and regression assumption testing.

Four candidate models were developed.

### Model 1 — Full Regression Model

The initial model incorporated all available predictors.

After transformation and residual analysis:

* R² ≈ **0.74**
* 14 predictors were statistically significant
* Significant multicollinearity remained among several predictors

### Model 2 — Backward Elimination

Backward elimination and stepwise selection were used to remove statistically insignificant predictors.

The resulting model contained **14 significant predictors** while retaining approximately the same explanatory power as the full model.

### Model 3 — Low-VIF Model

To reduce multicollinearity, predictors with high Variance Inflation Factor values were removed.

Highly related property variables such as bedrooms, bathrooms, and maximum guests were examined, with **Max Guests** retained as the representative variable.

The resulting model contained:

* 10 predictors
* Low multicollinearity
* Statistically significant predictors
* Adjusted R² ≈ **0.693**

### Model 4 — LASSO Regression

LASSO regularization was used for feature selection and to reduce the effects of multicollinearity.

The resulting model contained:

* 8 predictors
* Low multicollinearity
* Statistically significant predictors
* Adjusted R² ≈ **0.691**

This produced the simplest model while retaining much of the predictive ability of the larger models.

## Model Validation

The models were evaluated using a **70/30 train-test split**.

| Model                |   Test R² |      MSE |     RMSE | Predictors |
| -------------------- | --------: | -------: | -------: | ---------: |
| Full Model           |     0.733 |     8.05 |     2.84 |         16 |
| Backward Elimination | **0.733** | **8.05** | **2.84** |         14 |
| Low-VIF Model        |     0.685 |     9.50 |     3.08 |         10 |
| LASSO                |     0.683 |     9.55 |     3.09 |          8 |

Cross-validation produced similar results:

| Model                |   Mean R² | Mean MSE |
| -------------------- | --------: | -------: |
| Full Model           |     0.741 |     8.33 |
| Backward Elimination | **0.741** | **8.33** |
| Low-VIF Model        |     0.695 |     9.83 |
| LASSO                |     0.692 |     9.90 |

### Final Revenue Model

The **Backward Elimination model (Model 2)** was selected as the final revenue model.

It achieved the strongest predictive performance while removing unnecessary predictors from the full model.

The comparison also demonstrated a useful modeling trade-off:

* **Model 2:** stronger predictive performance but greater multicollinearity.
* **Model 4:** simpler and more stable model with slightly lower predictive performance.

## Superhost Classification

A logistic regression model was developed to predict:

`Airbnb Superhost`

The model was analyzed using both **Minitab** and **Python (`statsmodels`)**.

Predictors were evaluated using:

* Regression coefficients
* p-values
* Odds ratios
* Confidence intervals
* VIF
* ROC analysis
* Confusion matrix

The Python model identified relationships between Superhost status and features including:

| Predictor           | Odds Ratio |
| ------------------- | ---------: |
| Number of Reviews   |     1.0313 |
| Cancellation Policy |     0.8766 |
| Cleaning Fee        |     1.0032 |
| Minimum Stay        |     0.9720 |
| Pets Allowed        |     0.7261 |
| Instantbook Enabled |     0.2813 |
| Occupancy Rate LTM  |     0.9933 |
| Average Daily Rate  |     0.9993 |

The model achieved an **ROC AUC of approximately 0.77**, indicating moderate ability to distinguish Superhost from non-Superhost listings.

The confusion matrix also revealed a major limitation: although the model classified non-Superhosts effectively, it produced a relatively high number of false negatives when identifying Superhosts.

The overall reported misclassification rate was approximately **22.66%**.

## Key Findings

The analysis suggests several relationships within Dubai's Airbnb market:

* Entire homes generally command higher daily rates and annual revenue than individual rooms.
* Superhost listings are associated with higher ratings, more reviews, and higher annual revenue.
* Cancellation policy, listing type, and Superhost status show statistically significant relationships with listing performance.
* Property capacity variables such as bedrooms, bathrooms, and maximum guests are strongly correlated.
* Number of bookings has a moderate positive relationship with annual revenue.
* More complex regression models improved predictive performance but introduced multicollinearity.
* LASSO provided a simpler alternative with only a modest reduction in predictive performance.
* Logistic regression could distinguish Superhost status moderately well overall, but struggled to correctly identify many positive Superhost cases.

## Technologies & Methods

**Languages & Software**

* Python
* Jupyter Notebook
* Minitab

**Python ecosystem**

* pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* statsmodels
* scikit-learn

**Statistical & Machine Learning Methods**

* Exploratory Data Analysis
* Data preprocessing
* Correlation analysis
* Confidence intervals
* Hypothesis testing
* Chi-square tests
* F-tests
* One-way and two-way ANOVA
* Bonferroni multiple comparisons
* Multiple linear regression
* Box-Cox transformation
* Backward elimination
* Stepwise feature selection
* Variance Inflation Factor analysis
* LASSO regression
* Logistic regression
* Odds-ratio analysis
* Residual diagnostics
* Train-test validation
* Cross-validation
* ROC/AUC analysis
* Confusion matrix evaluation

## Repository Structure

```text
.
├── STA301_Project.ipynb
├── Project_Report.docx
├── Minitab_Project.mpx
└── README.md
```

* **`STA301_Project.ipynb`** — Python implementation covering data preparation, EDA, statistical analysis, model development, and validation.
* **`Project_Report.docx`** — Full project methodology, statistical interpretation, results, and discussion.
* **`Minitab_Project.mpx`** — Minitab analyses used for inferential statistics and logistic regression.
* **`README.md`** — Project overview and summary.

## Limitations & Future Work

Several limitations identified during the project provide opportunities for further development.

The revenue models continued to exhibit heteroscedasticity despite transformations and model refinement. Multicollinearity also required a trade-off between predictive performance and model simplicity.

For Superhost classification, class imbalance contributed to poor sensitivity and a relatively large number of false negatives.

Future work could therefore explore:

* Robust or weighted regression techniques for heteroscedasticity
* Alternative transformations and nonlinear models
* Tree-based regression methods such as Random Forest and Gradient Boosting
* Improved treatment of class imbalance for Superhost prediction
* Precision, recall, and F1-score optimization rather than relying primarily on overall accuracy
* Hyperparameter tuning and expanded cross-validation
* Additional location, property, and amenity features excluded from the current analysis

## Authors

Developed as part of the **STA 301 – Foundations of Statistics for Data Science** course at the American University of Sharjah.

* Mohamad Chehab
* Abdullah Salmeh
* Mustafa Alani

## Disclaimer

This project was developed for academic and educational purposes. The findings represent statistical relationships within the analyzed dataset and should not be interpreted as causal relationships or financial advice.

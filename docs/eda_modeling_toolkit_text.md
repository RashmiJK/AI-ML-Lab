# Beginner's Guide to Machine Learning Modeling: A Toolkit

Starting with machine learning can seem overwhelming, but this beginner-friendly framework—developed from insights gained in the second module of my Post Graduate Program in AI & ML journey—will make it easier to tackle problems and build your first models.

## Data Handling & Exploration

### Data Overview
**Key Concepts:** Basic data exploration and manipulation

**Details:** 
- Import NumPy and Pandas
- Essential methods: `.read_csv()`, `.head()`, `.tail()`, `.sample()`, `.shape`, `.copy()`, `.info()`, `.describe().T`, `.isnull().sum()`, `.duplicated().sum()`, `.nunique()`, `.unique()`, `.value_counts()`, `.groupby()`, `.drop()`

---

## Data Preprocessing

### Missing Value Treatment
**Key Concepts:** Techniques for handling missing data
- Use `.replace()` method for data cleaning

### Outlier Detection/Treatment
**Key Concepts:** Strategies for dealing with outliers
- Use `.quantile(0.25)` and `.quantile(0.75)` for quartile analysis

---

## Visualization

### Univariate Analysis
**Key Concepts:** Single variable visualization
- Import seaborn
- **Numerical data:** Histogram (`sns.histplot()`), Boxplot (`sns.boxplot()`, `df.boxplot(by='column')`)
- **Categorical data:** Countplot

### Bivariate Analysis
**Key Concepts:** Two variable relationships
- **Numerical-Numerical:** Scatter (`sns.scatterplot()`, `px.scatter_3d()`), Pair plots (`sns.pairplot()`), Line (`plt.plot()`), Heatmap (`sns.heatmap()`), Joint, Violin
- **Numerical-Categorical:** Line, Catplot, Boxplot
- **Categorical-Categorical:** Count, Box, Crosstab + stacked bar chart

### Multivariate Analysis
**Key Concepts:** Multiple variable relationships
- Use Catplot for complex multi-dimensional analysis

---

## Modeling

### Linear Regression
**Type:** Supervised Learning - Predicts numerical targets

**Key Concepts:**
- Best fit line using method of least squares
- **Error Formula:** 
  $$Error = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- **Expanded Error Formula:** 
  $$Error = \frac{1}{n} \sum_{i=1}^{n} (y_i - (b_0 + b_1 x_i))^2$$
- "Coefficients" and "intercept" are model parameters
- Multiple linear regression fits hyperplane in higher dimensions

**Evaluation Metrics:**
- **Mean Absolute Error (MAE):** 
  $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- **Root Mean Square Error (RMSE):** 
  $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- **Mean Absolute Percentage Error (MAPE):** 
  $$MAPE = \frac{1}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100$$
- **R-squared:** 
  $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
- **Adjusted R-squared:** 
  $$\bar{R}^2 = 1 - \left(1 - R^2\right) \frac{n - 1}{n - k - 1}$$
- Watch out for underfitting and overfitting

**Implementation Steps:**
- Prepare data with label encoding and one-hot encoding
- Use `train_test_split()` for data splitting
- Apply `LinearRegression()` model
- Build models for single feature, combinations, and all features
- Evaluate using `r2_score()` and `mean_absolute_error()`
- Select best model based on metrics

### Decision Tree
**Type:** Supervised Learning - Classification (can predict both categorical & numerical targets)

**Key Concepts:**
- Uses Gini impurity (0 = pure, 0.5 = high impurity) and Entropy
- Tree-based decision making through splits

**Evaluation Metrics:**
- **Confusion Matrix** for detailed performance analysis
- **Accuracy:** 
  $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
  *(Note: Use other metrics for imbalanced data)*
- **Recall (Sensitivity):** 
  $$Recall = \frac{TP}{TP + FN}$$
- **Precision:** 
  $$Precision = \frac{TP}{TP + FP}$$
- **F1 Score:** 
  $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Model Optimization:**
- **Default Fit:** Basic tree without constraints
- **Pre-pruning:** Hyperparameter tuning using GridSearchCV
- **Post-pruning:** Cost complexity pruning with ccp_alpha
  $$\alpha = \frac{Error(Pruned\ Tree) - Error(Original\ Tree)}{Number\ of\ nodes\ reduced}$$

**Implementation Steps:**
- Prepare data using `pd.get_dummies()` for nominal categorical values
- Use appropriate encoding for ordinal categorical values
- Split data with `train_test_split()`
- Apply `DecisionTreeClassifier()` with hyperparameters (class_weight, max_depth, max_leaf_nodes, min_samples_split, etc.)
- Evaluate using `confusion_matrix()`, `f1_score()`, `accuracy_score()`, `recall_score()`, `precision_score()`
- Use `cost_complexity_pruning_path()` for post-pruning
- Select best model based on suitable metrics

### Clustering
**Type:** Unsupervised Learning - Pattern Discovery

**Key Concepts:**
- **Distance Metrics:** Euclidean and Manhattan distance
- **K-Means Clustering:** Uses centroids to group data points

**Choosing Optimal K:**
- **Elbow Method:** Uses Within-Cluster Sum of Squares (WCSS)
  $$WCSS = \sum_{j=1}^{k} \sum_{x_i \in C_j} (x_i - \overline{x_j})^2$$
- **Silhouette Score:** Measures cluster cohesion and separation
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

**Dimensionality Reduction:**
- **t-SNE:** For visualization with perplexity parameter tuning

**Implementation Steps:**

**Extended EDA + Visualization:**
- Select only numeric data types for clustering
- **Crucial Step:** Scale numerical features using `StandardScaler()` for z-score scaling

**Dimensionality Reduction:**
- Use `TSNE()` for 2D or 3D visualization
- Experiment with different perplexity values
- Visualize using scatter plots to suggest optimal K value

**Clustering Process:**
- Apply `KMeans()` algorithm
- Use `.inertia_` to get WCSS metric for elbow method plot
- Compute Silhouette score and create visualization
- Reassess optimal K value based on both methods
- Perform cluster profiling by assigning cluster labels to data rows

---

## Summary
This toolkit provides a comprehensive foundation for tackling machine learning problems. Remember to always start with thorough data exploration, choose appropriate preprocessing techniques, and select evaluation metrics that align with your problem type and business objectives.
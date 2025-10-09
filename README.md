# linear-regression-housing
This project predicts housing prices using a Linear Regression model trained on features such as area, number of bedrooms, bathrooms, and amenities. The dataset (Housing.csv) contains both numerical and categorical features, with categorical values encoded using pd.get_dummies(). The workflow involves encoding categorical variables, splitting the data into 80% training and 20% testing sets, training the model, and evaluating it using MAE, MSE, and R² score. The model achieves an R² of approximately 0.65, with an MAE of ~₹9.70×10⁵ and MSE of 1.75×10¹². Analysis of feature coefficients shows that larger area, more bathrooms, and air conditioning significantly increase prices, while unfurnished homes tend to lower them. A scatter plot comparing actual and predicted prices shows a close alignment near the ideal diagonal, indicating reasonable accuracy. This model serves as a strong baseline, with potential for improvement through additional features and more advanced regression techniques.

Overview
Predict house prices using Linear Regression based on features like area, bedrooms, bathrooms, and amenities. The model explains ~65% of the price variance.

Dataset
-Housing.csv with numerical and categorical features (encoded using pd.get_dummies).

Installation
pip install pandas numpy scikit-learn matplotlib

Workflow

Encode categorical variables with get_dummies.
Split data into 80% train, 20% test.
Train Linear Regression model.
Evaluate with MAE, MSE, and R².
Plot Actual vs Predicted prices and interpret coefficients.
Results

Metric	Value
MAE	~970,043
MSE	1.75e+12
R² Score	0.65
Feature Impact (Selected)

Feature	Coefficient (₹)
Area	+235.97 / sq.ft
Bathrooms	+1,094,444.79
Airconditioning_yes	+791,426.74
Furnishingstatus_unfurnished	–413,645.06
Visualization
Scatter plot of actual vs predicted prices shows good alignment near the ideal line.

Conclusion
Model provides a solid baseline. Baths, AC, and area positively impact price. Furnishing affects price negatively if semi/unfurnished.

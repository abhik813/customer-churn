I downloaded the dataset provided by Sunbase and began with data preprocessing, addressing issues like identifying null values and outliers within the dataset. Subsequently, I carried out essential data visualization tasks, uncovering several key findings:

1. The frequencies of churn and non-churn instances are nearly balanced.
2. There is a limited presence of outliers in the dataset.

Following this, I proceeded with feature engineering, recognizing that the total amount charged to date could be a crucial feature. Consequently, I included "total amount charged" as one of the features.

Next, I divided the dataset into numerical and categorical columns, performing scaling, normalization, and one-hot encoding for the categorical variables.

For model training, I utilized various algorithms, including Logistic Regression, RandomForestClassifier, Support Vector Machine, and neural networks. Additionally, I fine-tuned the hyperparameters to enhance model performance.

Lastly, I saved the trained model using h5py and deployed it in a production-like environment. Our model is capable of accepting new input features provided by users and making predictions accordingly.

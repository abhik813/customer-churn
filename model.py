import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder

churn_data = pd.read_csv("./customer_churn_large_dataset (1).csv")

n_rows = churn_data.shape[0]
n_cols = churn_data.shape[1]

numeric_data = churn_data.select_dtypes(include=[np.number])
categorical_data = churn_data.select_dtypes(exclude=[np.number])

numeric_data.hist(figsize=(11,10))

total_charged_amount = churn_data['Subscription_Length_Months'] * churn_data['Monthly_Bill']

churn_data.insert(3,"Total_Amount_Charged",total_charged_amount)
input_cols = list(churn_data.columns[2:-1])
input_cols
output_cols = 'Churn'

input_df = churn_data[input_cols].copy()
output_df =churn_data[output_cols].copy()

numeric_input_data = input_df.select_dtypes(include=[np.number])
categorical_input_data = input_df.select_dtypes(exclude=[np.number])

numeric_data = churn_data.select_dtypes(include=[np.number])
categorical_data = churn_data.select_dtypes(exclude=[np.number])
numeric_data.corr(method='spearman')

from sklearn.preprocessing import MaxAbsScaler

cols_scaled = ['Age' , 'Subscription_Length_Months' , 'Monthly_Bill' , 'Total_Usage_GB','Total_Amount_Charged']
categorical_cols = ['Gender' , 'Location']

minscaler = MaxAbsScaler()
minscaler = MaxAbsScaler().fit(input_df[cols_scaled])

input_df[cols_scaled] = minscaler.transform(input_df[cols_scaled])

encoder = OneHotEncoder(sparse=False)
encoder.fit(input_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
input_df.drop(['Gender'] , axis=1 , inplace=True)
input_df.drop(['Location'] , axis=1 , inplace=True)

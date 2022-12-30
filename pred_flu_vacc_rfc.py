
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f_train = pd.read_csv('./data/training_set_features.csv', index_col=0)
l_train = pd.read_csv('./data/training_set_labels.csv', index_col=0)

j_train = f_train.join(l_train)


#%%%%%%%%%%%% --this is to add to the jupyter notebook
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score

numeric_cols = f_train.columns[f_train.dtypes != "object"].values


model_1 = LogisticRegression()
model_2 = XGBClassifier()
model_3 = RandomForestClassifier()


numeric_preprocessing_steps = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('simple_imputer', SimpleImputer(strategy='median'))
])


preprocessor = ColumnTransformer(
    transformers = [
        ("numeric", numeric_preprocessing_steps, numeric_cols)
    ],
    remainder = "drop"
)



estimators = MultiOutputClassifier(
    VotingClassifier(estimators=[('lr', model_1), ('xgb', model_2), ('rf', model_3)], voting ='soft')
)



full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimators", estimators),
])




X_train, X_eval, y_train, y_eval = train_test_split(
                                                    f_train,
                                                    l_train,
                                                    test_size=0.33,
                                                    shuffle=True,
                                                    stratify=l_train,
                                                    random_state=101
                                                    )

full_pipeline.fit(X_train, y_train)
print(full_pipeline.score(X_train, y_train))
preds = full_pipeline.predict_proba(X_eval)


y_preds = pd.DataFrame(
    {
        "h1n1_vaccine": preds[0][:, 1],
        "seasonal_vaccine": preds[1][:, 1],
    },
    index = y_eval.index
)

print(roc_auc_score(y_eval, y_preds))
y_preds = (y_preds > 0.5)
print(accuracy_score(y_eval, y_preds))

full_pipeline.fit(f_train, l_train)
None


f_test = pd.read_csv('./data/test_set_features.csv', index_col=0)
test_probas = full_pipeline.predict_proba(f_test)
submission_df = pd.read_csv('./data/submission_format.csv', index_col=0)

np.testing.assert_array_equal(f_test.index.values,
                              submission_df.index.values)

# Save predictions to submission data frame
submission_df["h1n1_vaccine"] = test_probas[0][:, 1]
submission_df["seasonal_vaccine"] = test_probas[1][:, 1]
print(submission_df.head())
submission_df.to_csv('my_submission.csv', index=True)

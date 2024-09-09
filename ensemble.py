# Created by trilo at 28-08-2024
import streamlit as st

votting_classification = st.Page("votting/voting_classifier.py", title="Classification", icon=":material/category:")
# regression = st.Page("regression/voting_regressor.py", title="Regression", icon=":material/trending_up:")
bagging_classification = st.Page("bagging/bagging_classifier.py", title="Classification", icon=":material/category:")
bagging_regression = st.Page("bagging/bagging_regressor.py", title="Regression", icon=":material/trending_up:")
rf_classification = st.Page("random_forest/random_forest_classifier.py", title="Classification", icon=":material/category:")

readme = st.Page("readme/readme.py", title="Readme", icon=":material/dashboard:")

pg = st.navigation(
    {
        "Voting Model": [votting_classification],
        "Bagging Model": [bagging_classification, bagging_regression],
        "Random Forest": [rf_classification],
        # "README": [readme],
    }
)

pg.run()

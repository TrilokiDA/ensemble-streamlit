# Created by trilo at 26-08-2024
import streamlit as st
import matplotlib.pyplot as plt
from votting.dataset_fun import load_initial_graph
from votting.model_fun import modelClass

st.sidebar.markdown("## :rainbow[Voting Classifier]")
# Dataset selction dropdown
dataset = st.sidebar.selectbox(
    "Dataset",
    ("U-Shaped", "Linearly Separable", "Outlier", "Two Spirals", "Concentric Circles", "XOR")
)
# multi select estimators
estimators = st.sidebar.multiselect(
    'Estimators',
    ['KNN', 'Logistic Regression', 'Gaussian Naive Bayes', 'SVM', 'Random Forest']
)
#
voting_type = st.sidebar.radio(
    "Voting Type",
    ('hard', 'soft')
)

st.header(dataset)
fig, ax = plt.subplots()

# Plot initial graph
df = load_initial_graph(dataset, ax)
orig = st.pyplot(fig)

# Extract X and Y
X = df.iloc[:, :2].values
y = df.iloc[:, -1].values

# Create sthelper object
modelfun = modelClass(X, y)

# Button
if st.sidebar.button("Run"):
    models = modelfun.create_base_estimators(estimators, voting_type)
    voting_clf, voting_clf_accuracy = modelfun.train_voting_classifier(models, voting_type)
    modelfun.draw_main_graph(voting_clf, ax)
    orig.pyplot(fig)
    figs = modelfun.plot_other_graphs(models)

    # plot accuracy
    st.sidebar.header("Classification Metrics")
    st.sidebar.text("Voting Classifier accuracy:" + str(voting_clf_accuracy))

    accuracies = modelfun.calculate_base_model_accuracy(models)

    for i in range(len(accuracies)):
        st.sidebar.text("Accuracy for Model " + str(i + 1) + " - " + str(accuracies[i]))

    counter = 0
    for i in st.columns(len(figs)):
        with i:
            st.pyplot(figs[counter])
            st.text(counter)
        counter += 1
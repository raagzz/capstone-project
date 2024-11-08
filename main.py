import pickle
import json

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from uml_utils import *


st.set_page_config(layout="wide", page_title="ALU Platform")

# Dictionary of available estimators
ESTIMATORS = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Dictionary of query strategies
QUERY_STRATEGIES = {
    'Uncertainty Sampling': uncertainty_sampling,
    'Margin Sampling': margin_sampling,
    'Entropy Sampling': entropy_sampling
}


def initialize_session_state():
    """Initialize session state variables"""
    if 'accuracy_scores' not in st.session_state:
        st.session_state.accuracy_scores = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = 0
    if 'learner' not in st.session_state:
        st.session_state.learner = None
    if 'query_strategy' not in st.session_state:
        st.session_state.query_strategy = uncertainty_sampling
    if 'remaining_queries' not in st.session_state:
        st.session_state.remaining_queries = 0
    if 'label_column' not in st.session_state:
        st.session_state.label_column = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'labels' not in st.session_state:
        st.session_state.labels = []

def preview_data(data, label_column=None):
    """Display a preview of the uploaded data"""
    st.info(f"Number of rows: {len(data)}", icon=":material/table_rows:")
    st.info(f"Number of columns: {len(data.columns)}", icon=":material/view_column:")
    st.write("### Data Preview")
    st.write(data.head())


def process_uploaded_file(uploaded_file, label_column=None):
    """Process uploaded CSV file and return features and labels if present"""
    data = pd.read_csv(uploaded_file)

    # Store feature names
    st.session_state.feature_names = data.columns.tolist()

    # Preview the data
    preview_data(data, label_column)

    # Check if label column exists and process accordingly
    if label_column and label_column in data.columns:
        X = data.drop(label_column, axis=1)
        y = data[label_column]
        le = LabelEncoder()
        y = le.fit_transform(y)
        st.session_state.feature_names.remove(label_column)
    else:
        X = data
        y = None

    return X.values, y


def reset_session():
    """Reset all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()


def update_plot():
    """Update the accuracy plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(st.session_state.accuracy_scores))),
        y=st.session_state.accuracy_scores,
        mode='lines+markers',
        name='Accuracy'
    ))

    fig.update_layout(
        title='Model Accuracy Over Queries',
        xaxis_title='Number of Queries',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1]),
        showlegend=True
    )
    return fig

def display_metric(name, before_value, after_value):
    if before_value is None or after_value is None:
        st.metric(name, after_value if after_value is not None else "N/A")
        return
    
    delta = after_value - before_value
    delta_str = f"{delta:.2%}" if delta != 0 else "No Change"
    delta_color = "green" if delta > 0 else "red" if delta < 0 else "off"
    
    st.metric(name, f"{after_value:.2%}", delta_str, delta_color)

def calculate_delta(before, after):
    if before is not None and after is not None:
        return round(after - before, 3)
    return 'N/A'


def main():
    st.title('ALU: Active Learning Meets Machine Unlearning')

    # Initialize session state
    initialize_session_state()

    # Sidebar configuration
    st.sidebar.header('ALU Configurations')
    st.sidebar.subheader('Active Learning')

    # Model selection
    selected_model = st.sidebar.selectbox(
        'Select Classifier',
        list(ESTIMATORS.keys())
    )

    # Query strategy selection
    selected_strategy = st.sidebar.selectbox(
        'Select Query Strategy',
        list(QUERY_STRATEGIES.keys())
    )

    # Number of queries
    n_queries = st.sidebar.number_input('Number of Queries', 1, 100, 10)

    # Files uploader
    st.header('Data Upload')

    col1, col2 = st.columns(2)
    with col1:
        # Labeled data upload section
        st.subheader("Upload Labeled Dataset")
        labeled_file = st.file_uploader("Upload Labeled Dataset (CSV)", type=['csv'], key='labeled')

    # Label column selection
    label_column = None
    if labeled_file:
        # Read the CSV to get column names
        df_preview = pd.read_csv(labeled_file)
        label_column = st.selectbox(
            "Select the Label Column",
            options=df_preview.columns.tolist(),
            key='label_column_key'
        )
        # Reset file pointer
        labeled_file.seek(0)

    with col2:
        # Unlabeled data upload section
        st.subheader("Upload Unlabeled Dataset")
        unlabeled_file = st.file_uploader(
            "Upload Unlabeled Dataset (CSV)",
            type=['csv'],
            key='unlabeled'
        )

    # Process uploaded files and initialize learner
    if labeled_file and unlabeled_file and label_column and st.button('Start Active Learning'):
        with st.spinner('Processing datasets...'):
            try:

                col1, col2 = st.columns(2)
                with col1:
                    st.header("Labeled Dataset Info")
                    # Process labeled data
                    X_labeled, y_labeled = process_uploaded_file(labeled_file, label_column)
                    st.session_state.labels = y_labeled

                with col2:
                    st.header("Unlabeled Dataset Info")
                    # Process unlabeled data
                    X_unlabeled, _ = process_uploaded_file(unlabeled_file)

                st.divider()

                # Validate feature consistency
                if X_labeled.shape[1] != X_unlabeled.shape[1]:
                    st.error("Error: The number of features in labeled and unlabeled datasets must match!")
                    return

                # Split labeled data into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X_labeled, y_labeled,
                    test_size=0.5,
                    random_state=42
                )

                # Initialize active learner
                st.session_state.learner = ActiveLearner(
                    estimator=ESTIMATORS[selected_model],
                    query_strategy=QUERY_STRATEGIES[selected_strategy],
                    X_training=X_train,
                    y_training=y_train
                )

                # Store necessary variables in session state
                st.session_state.X_pool = X_unlabeled
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.remaining_queries = n_queries
                st.session_state.accuracy_scores = []
                st.session_state.current_query = 0
                st.session_state.label_column = label_column

                st.success('Active learner initialized successfully!')
            except Exception as e:
                st.error(f"Error initializing active learner: {str(e)}")


    # Active learning loop
    if st.session_state.learner is not None and st.session_state.remaining_queries > 0:
        st.header('Active Learning')

        # Get next query instance
        query_idx, query_inst = st.session_state.learner.query(
            st.session_state.X_pool,
            n_instances=1
        )

        # Display instance features
        st.subheader("Annotate Label for Queries")
        query_dict = {}
        for i, value in enumerate(query_inst[0]):
            feature_name = st.session_state.feature_names[i] if st.session_state.feature_names else f"Feature {i + 1}"
            query_dict[feature_name] = value

        st.dataframe(pd.DataFrame(query_dict, index=['Query Sample']))


        # Get user input
        label = st.radio(
            "Select label for this instance:",
            options=np.unique(st.session_state.labels),
            key=f"query_{st.session_state.current_query}"
        )

        if st.button('Submit Label'):
            # Teach the model
            st.session_state.learner.teach(
                query_inst.reshape(1, -1),
                np.array([label], dtype=int)
            )

            # Update pool
            st.session_state.X_pool = np.delete(
                st.session_state.X_pool,
                query_idx,
                axis=0
            )

            # Calculate and store accuracy
            accuracy = st.session_state.learner.score(
                st.session_state.X_test,
                st.session_state.y_test
            )

            st.session_state.accuracy_scores.append(accuracy)

            # Update counters
            st.session_state.current_query += 1
            st.session_state.remaining_queries -= 1

            # Force rerun to update the display
            st.rerun()


    if st.session_state.learner:
        st.plotly_chart(update_plot(), use_container_width=True)

    if st.session_state.learner:
        st.download_button("Download this Model", pickle.dumps(st.session_state.learner.estimator), 'model.bin')

    # Display statistics in sidebar
    st.sidebar.subheader("Training Statistics")
    st.sidebar.metric("Queries Completed", len(st.session_state.accuracy_scores))

    if st.session_state.accuracy_scores:

        acc_list = st.session_state.accuracy_scores
        col1, col2 = st.columns(2)

        with col1:
            st.sidebar.metric("Current Accuracy", np.round(acc_list[-1], 4),
                          delta=np.round(acc_list[-1] - acc_list[-2], 2)
                          if st.session_state.current_query > 1 else 0)

        with col2:
            st.sidebar.metric("Best Accuracy", np.round(max(acc_list), 4))

    # Reset button
    if st.sidebar.button('Reset Application'):
        reset_session()
        st.rerun()

    if st.session_state.learner and st.session_state.remaining_queries == 0:
        st.header("Machine Unlearning")
        uploaded_file_unlearn = st.file_uploader("Upload a Data for Unlearning", type="csv")
        if uploaded_file_unlearn:
            data_unlearning = pd.read_csv(uploaded_file_unlearn)
            targeted_col = label_column
            features = [col for col in data_unlearning.columns if col != targeted_col]
            X = data_unlearning[features]
            y = data_unlearning[targeted_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model_name = st.selectbox("Select Model", SISA.AVAILABLE_MODELS.keys())
            n_shards = st.slider("Number of Shards", 1, 10, 5)
            n_estimators = st.slider("Number of Estimators (for ensemble models)", 1, 100, 10)
            selection_strategy = st.selectbox("Unlearning Strategy", ["random", "feature_based"])

            strategy_params = None
            if selection_strategy == "random":
                strategy_params = {"n_samples": st.number_input("Number of Samples to Forget", 1, 100, 10)}
            elif selection_strategy == "feature_based":
                feature = st.selectbox("Feature", X_train.columns)
                operator = st.selectbox("Operator", ["gt", "lt", "eq", "between"])
                value = st.text_input("Value")
                if operator == "between":
                    try:
                        value = json.loads(value)
                        assert len(value) == 2
                    except:
                        st.write("Enter two values in the format: [min, max]")
                strategy_params = {'conditions':[{'feature': feature, 'operator': operator, 'value': int(value)}]}

            if st.button("Run Expriment"):
                report = run_experiment(X_train, X_test, y_train, y_test, model_name=model_name, selection_strategy=selection_strategy, strategy_params=strategy_params, n_shards=n_shards, n_estimators=n_estimators)
                st.success("Machine Unlearning Successful")
                st.write("Experiment Report")
                st.download_button("Download this Report", json.dumps(report), 'report.json')
                #st.json(report)

                #print(report['metrics_before_unlearning']['accuracy'])
                #print(report)

                

                # Displaying metrics with delta in a two-column layout within each expander
                with st.expander("Before Unlearning Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", report['metrics_before_unlearning'].get('accuracy', 'N/A'))
                        st.metric("Recall", report['metrics_before_unlearning'].get('recall', 'N/A'))
                    with col2:
                        st.metric("Precision", report['metrics_before_unlearning'].get('precision', 'N/A'))
                        st.metric("F1 Score", report['metrics_before_unlearning'].get('f1', 'N/A'))

                with st.expander("After Unlearning Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", report['metrics_after_unlearning'].get('accuracy', 'N/A'),
                                delta=calculate_delta(report['metrics_before_unlearning'].get('accuracy'),
                                                        report['metrics_after_unlearning'].get('accuracy')))
                        st.metric("Recall", report['metrics_after_unlearning'].get('recall', 'N/A'),
                                delta=calculate_delta(report['metrics_before_unlearning'].get('recall'),
                                                        report['metrics_after_unlearning'].get('recall')))
                    with col2:
                        st.metric("Precision", report['metrics_after_unlearning'].get('precision', 'N/A'),
                                delta=calculate_delta(report['metrics_before_unlearning'].get('precision'),
                                                        report['metrics_after_unlearning'].get('precision')))
                        st.metric("F1 Score", report['metrics_after_unlearning'].get('f1', 'N/A'),
                                delta=calculate_delta(report['metrics_before_unlearning'].get('f1'),
                                                        report['metrics_after_unlearning'].get('f1')))

                with st.expander("Experiment Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Name", report['experiment_details'].get('model_name', 'N/A'))
                        st.metric("Samples Forgotten", report['experiment_details'].get('samples_forgotten', 'N/A'))
                    with col2:
                        st.metric("Affected Shards", report['experiment_details'].get('affected_shards', 'N/A'))
                

                # st.sidebar.header("Machine Unlearning Metrics")
                # st.sidebar.subheader("Before Unlearning Metrics")
                # st.sidebar.metric("Accuracy",report['metrics_before_unlearning']['accuracy'])
                # st.sidebar.metric("Precision",report['metrics_before_unlearning']['precision'])
                # st.sidebar.metric("Recall",report['metrics_before_unlearning']['recall'])
                # st.sidebar.metric("F1",report['metrics_before_unlearning']['f1'])
                
                # st.sidebar.subheader("After Unlearning Metrics")
                # st.sidebar.metric("Accuracy",report['metrics_after_unlearning']['accuracy'])
                # st.sidebar.metric("Precision",report['metrics_after_unlearning']['precision'])
                # st.sidebar.metric("Recall",report['metrics_after_unlearning']['recall'])
                # st.sidebar.metric("F1",report['metrics_after_unlearning']['f1'])

                # st.sidebar.subheader("Expriment Deatils")
                # st.sidebar.metric("Model Name",report['experiment_details']['model_name'])
                # st.sidebar.metric("Samples Forgotten",report['experiment_details']['samples_forgotten'])
                # st.sidebar.metric("Affected Shards",report['experiment_details']['affected_shards'])

if __name__ == "__main__":
    main()
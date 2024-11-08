 import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling, margin_sampling

# Initialize models dictionary
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}


@st.cache_data
def data_preprocessing(df, target):
    """
    Preprocess the data with caching for better performance
    """
    X = df.drop(target, axis=1)
    y = df[target].str.lower().values if df[target].dtype == 'object' else df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, y_train, X_test, y_test


def reset_session_state():
    """
    Reset all session state variables
    """
    keys_to_remove = ['learner', 'X_train', 'y_train', 'current_query',
                      'initial_score', 'scores', 'form1', 'form2']
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]


# Page configuration
st.set_page_config(layout="wide", page_title="Active Learning Platform")

# Add CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #dcffe4;
        color: #0c5460;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Learn and Unlearn: Active Learning and Machine Unlearning No-Code Platform")

# Add a reset button at the top
if st.button("Reset Application"):
    reset_session_state()
    st.rerun()

# File uploaders with error handling
try:
    data = st.file_uploader('Upload your Labeled Data (CSV)', type=['csv'])
    unlabeled_data = st.file_uploader("Upload your Unlabeled Data (CSV)", type=['csv'])
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

if data is not None:
    try:
        dataframe = pd.read_csv(data)

        with st.form('options_menu'):
            st.subheader('Select your Preferences')
            col1, col2 = st.columns(2)

            with col1:
                target = st.selectbox("Select the Target Feature",
                                      dataframe.columns,
                                      help="Choose the column you want to predict")

            with col2:
                model = st.selectbox("Select your ML model",
                                     list(models.keys()),
                                     help="Choose the machine learning algorithm")

            submit = st.form_submit_button(label='Submit')

            if not submit and not st.session_state.get('form1'):
                st.stop()

            st.session_state['form1'] = True

        # Process data and display dataset info
        X_train, y_train, X_test, y_test = data_preprocessing(dataframe, target)
        labels = np.unique(y_train)

        col1, col2 = st.columns(2)
        with col1:
            st.header('Dataset Preview')
            st.dataframe(dataframe.head(), use_container_width=True)
        with col2:
            st.header('Dataset Info')
            st.write(f"Total samples: {len(dataframe)}")
            st.write(f"Features: {dataframe.drop(target, axis=1).columns.tolist()}")
            st.write(f"Number of classes: {len(labels)}")
            st.write(f"Classes: {labels.tolist()}")

        st.subheader('Active Learning')
        if unlabeled_data is not None:
            try:
                X_pool = pd.read_csv(unlabeled_data).values
                max_queries = X_pool.shape[0]

                with st.form('active_learning'):
                    col1, col2 = st.columns(2)

                    with col1:
                        al_strategy = st.selectbox(
                            "Select Active Learning Strategy",
                            ['Uncertainty', 'Entropy', 'Margin'],
                            help="Choose the strategy for selecting new samples"
                        )

                    with col2:
                        queries = st.number_input(
                            "Number of queries:",
                            min_value=1,
                            max_value=max_queries,
                            value=min(5, max_queries),
                            help="Number of samples to label"
                        )

                    strategies = {
                        'Uncertainty': uncertainty_sampling,
                        'Entropy': entropy_sampling,
                        'Margin': margin_sampling
                    }

                    al_submit = st.form_submit_button(label='Start Active Learning')

                    if not al_submit and not st.session_state.get('form2'):
                        st.stop()
                    st.session_state['form2'] = True

                # Initialize or retrieve learner from session state
                if 'learner' not in st.session_state:
                    with st.spinner('Initializing model...'):
                        st.session_state.learner = ActiveLearner(
                            estimator=models[model],
                            query_strategy=strategies[al_strategy],
                            X_training=X_train,
                            y_training=y_train
                        )
                        st.session_state.learner.teach(X_train, y_train)
                        st.session_state.X_train = X_train.copy()
                        st.session_state.y_train = y_train.copy()
                        st.session_state.current_query = 0
                        st.session_state.initial_score = st.session_state.learner.score(X_test, y_test)
                        st.session_state.scores = [st.session_state.initial_score]

                # Create two columns for scores and progress
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Initial Score", f"{st.session_state.initial_score:.4f}")
                with col2:
                    st.metric("Progress", f"{st.session_state.current_query}/{queries} queries")

                # Active learning interface
                if st.session_state.current_query < queries:
                    query_idx, query_sample = st.session_state.learner.query(X_pool)

                    st.write("### Label New Sample")
                    query_label = st.selectbox(
                        "Select label for this sample:",
                        options=labels,
                        key=f"query_{st.session_state.current_query}",
                        help="Choose the correct label for this sample"
                    )

                    # Display sample features in a more readable format
                    st.write("Sample features:")
                    sample_df = pd.DataFrame(query_sample.reshape(1, -1),
                                             columns=dataframe.drop(target, axis=1).columns)
                    st.dataframe(sample_df, use_container_width=True)

                    if st.button("Submit Label", key=f"submit_{st.session_state.current_query}"):
                        with st.spinner('Updating model...'):
                            st.session_state.learner.teach(query_sample.reshape(1, -1), [query_label])
                            st.session_state.X_train = np.vstack([st.session_state.X_train, query_sample])
                            st.session_state.y_train = np.append(st.session_state.y_train, query_label)
                            current_score = st.session_state.learner.score(X_test, y_test)
                            st.session_state.scores.append(current_score)
                            st.session_state.current_query += 1
                            st.rerun()

                # Display scores history
                if len(st.session_state.scores) > 1:
                    st.write("### Learning Progress")
                    progress_df = pd.DataFrame({
                        'Iteration': range(len(st.session_state.scores)),
                        'Score': st.session_state.scores
                    })
                    st.line_chart(progress_df.set_index('Iteration'))

                # Show completion message
                if st.session_state.current_query >= queries:
                    st.success("Active learning process completed!")
                    st.write(f"Final Score: {st.session_state.scores[-1]:.4f}")
                    st.write(f"Total Improvement: {(st.session_state.scores[-1] - st.session_state.initial_score):.4f}")

            except Exception as e:
                st.error(f"Error in active learning process: {str(e)}")
                st.stop()

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()















# models = {'Logistic Regression': LogisticRegression(), 'SVM': SVC(), 'Decision Tree': DecisionTreeClassifier(),'Random Forest': RandomForestClassifier()}
#
# def data_preprocessing(df, target):
#     X = df.drop(target, axis=1)
#     y = df[target].str.lower().values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#
#     return X_train, y_train, X_test, y_test
#
#
# st.set_page_config(layout="wide")
# st.title("Learn and Unlearn: Active Learning and Machine Unlearning No-Code Platform")
#
# data = st.file_uploader('Upload your Labeled Data')
# unlabeled_data = st.file_uploader("Upload your Unlabeled Data")
#
# if data is not None:
#     dataframe = pd.read_csv(data)
#
#     with st.form('options_menu'):
#         st.subheader('Select your Preferences')
#         target = st.selectbox("Select the Target Feature", dataframe.columns)
#         model = st.selectbox("Select your ML model", ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest'])
#         submit = st.form_submit_button(label='Submit')
#
#         if not submit and not st.session_state.get('form1'):
#             st.stop()
#
#         st.session_state['form1'] = True
#
#     X_train, y_train, X_test, y_test = data_preprocessing(dataframe, target)
#     labels = np.unique(y_train)
#
#     st.header('Dataset')
#     st.dataframe(dataframe)
#
#     st.subheader('Active Learning')
#     if unlabeled_data is not None:
#         X_pool = pd.read_csv(unlabeled_data).values
#         max_queries = X_pool.shape[0]
#
#         with st.form('active_learning'):
#             al_strategy = st.selectbox("Select your Desired Active Learning Strategy",
#                                        ['Uncertainty', 'Entropy', 'Margin'])
#             strategies = {'Uncertainty': uncertainty_sampling,
#                           'Entropy': entropy_sampling,
#                           'Margin': margin_sampling}
#             queries = st.number_input("Enter the desired number of queries (No. of datapoints to Label):", 0,
#                                       max_queries)
#             al_submit = st.form_submit_button(label='Submit')
#
#             if not al_submit and not st.session_state.get('form2'):
#                 st.stop()
#             st.session_state['form2'] = True
#
#         # Initialize the model and session state variables after forms are submitted
#         if 'learner' not in st.session_state:
#             st.session_state.learner = ActiveLearner(
#                 estimator=models[model],
#                 query_strategy=strategies[al_strategy],
#                 X_training=X_train,
#                 y_training=y_train
#             )
#             st.session_state.learner.teach(X_train, y_train)
#             st.session_state.X_train = X_train.copy()
#             st.session_state.y_train = y_train.copy()
#             st.session_state.current_query = 0
#             st.session_state.initial_score = st.session_state.learner.score(X_test, y_test)
#             st.session_state.scores = [st.session_state.initial_score]
#
#         # Display initial score
#         st.write(f"Initial Score: {st.session_state.initial_score}")
#
#         # Only proceed if we haven't completed all queries
#         if st.session_state.current_query < queries:
#             # Get the next sample to label
#             query_idx, query_sample = st.session_state.learner.query(X_pool)
#
#             # Create the select box for labeling
#             label_key = f"query_{st.session_state.current_query}"
#             query_label = st.selectbox(
#                 f"Enter the label for this sample: {query_sample}:",
#                 options=labels,
#                 key=label_key
#             )
#
#             # Add a submit button for this label
#             if st.button("Submit Label", key=f"submit_{st.session_state.current_query}"):
#                 # Update the model with new labeled data
#                 st.session_state.learner.teach(query_sample.reshape(1, -1), [query_label])
#
#                 # Update training data
#                 st.session_state.X_train = np.vstack([st.session_state.X_train, query_sample])
#                 st.session_state.y_train = np.append(st.session_state.y_train, query_label)
#
#                 # Calculate and store new score
#                 current_score = st.session_state.learner.score(X_test, y_test)
#                 st.session_state.scores.append(current_score)
#
#                 # Increment query counter
#                 st.session_state.current_query += 1
#
#                 # Force a rerun to update the interface
#                 st.rerun()
#
#         # Display progress and scores
#         st.write(f"Completed {st.session_state.current_query} out of {queries} queries")
#
#         # Show all scores in a line
#         if len(st.session_state.scores) > 1:
#             st.write("### Scores History")
#             for i, score in enumerate(st.session_state.scores):
#                 st.write(f"Score after iteration {i}: {score}")
#
#         # Show completion message
#         if st.session_state.current_query >= queries:
#             st.success("Active learning process completed!")

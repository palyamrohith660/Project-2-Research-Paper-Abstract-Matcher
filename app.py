import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from feature_engine.selection import *
from FEATURE_SELECTION import *
from SELECTORS import *
from CREATORS import *

class Features:
    def __init__(self, data):
        self.dataset = data

    def display(self):
        with st.sidebar:
            selected_option = option_menu("Select stage", ["See Correlations", "Select Features", "Create Features"])
            
        if selected_option == "See Correlations":
            self.correlations()
        elif selected_option == "Select Features":
            self.select_features()
        elif selected_option == "Create Features":
            self.create_features()

    def correlations(self):
        tab1, tab2 = st.tabs(["Perform Operations", "View Data"])
        
        with tab1:
            col1, col2 = st.columns([1, 2],border=True)
            radio_options = col1.radio("Options Were", ["pearson", "spearman", "kendall", "point", "cramers"])
            
            with col2:
                select_columns = st.multiselect("Select columns", self.dataset.columns.tolist())
                if select_columns:
                    dataset = self.dataset.copy(deep=True)[select_columns]
                    if st.button("Execute Feature Selection", use_container_width=True):
                        feature_selection = FeatureSelection(dataset)
                        try:
                            getattr(feature_selection, radio_options, lambda: st.warning("Invalid Method"))()
                        except Exception as e:
                            st.error(e)
        
        with tab2:
            pass
    def create_features(self):
        tab1,tab2=st.tabs(["Perform Operations","View Data"])
        with tab1:
            value_dict={"math_features":math_features,"relative_features":relative_features,
                       "cyclical_features":cyclical_features,"descision_tree_features":decision_tree_features,
                       "custom_features":custom_features,"add columns from outputs":addColumnsFromOutputs}
            col1,col2=st.columns([1,2],border=True)
            radio_options=col1.radio("Select the option to perform",value_dict.keys())
            if radio_options:
                with col2:
                    value_dict[radio_options](radio_options,self.dataset)
    def select_features(self):
        feature_methods = {
            "Drop features": drop_features,
            "Drop Constant Features": drop_constant_features,
            "Drop Duplicated Features": drop_duplicated_features,
            "Drop Correlated Features": drop_correlated_features,
            "Smart Correlated Selection": smart_correlated_selection,
            "MRMR": mrmr_feature_selection,
            "Select By Single Feature Performance": select_by_single_feature_performance,
            "Recursive Feature Elimination": recursive_feature_elimination,
            "Recursive Feature Addition": recursive_feature_addition,
            "Drop High PSI Features": drop_high_psi_features,
            "Select By Information Value": select_by_information_value,
            "Select By Shuffling": select_by_shuffling,
            "Select By Target Mean Performance": select_by_target_mean_performance,
            "Probe Feature Selection": probe_feature_selection
        }
        
        tab1, tab2 = st.tabs(["Perform Operations", "View Data"])
        
        with tab1:
            col1, col2 = st.columns([1, 2],border=True)
            radio_options = col1.radio("Options Were", list(feature_methods.keys()))
            
            if radio_options:
                with col2:
                    feature_methods[radio_options](radio_options,self.dataset)
    

file_uploader = st.sidebar.file_uploader("Upload CSV", type=['csv'])
if file_uploader:
    dataframe = pd.read_csv(file_uploader)
    st.session_state['readed_csv'] = dataframe
    selected_output = st.selectbox("Outputs to select", [x for x in st.session_state.keys() if x])
    if selected_output:
        Features(st.session_state['readed_csv']).display()

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import chardet
import numpy_financial as npf

if "numeric_dataframe" not in st.session_state:
    st.session_state["numeric_dataframe"]=None
#Declaring the global calsses
class Descriptive:
    def __init__(self,dataset):
        self.dataset=dataset
    def parameter_setup(self,Type):
        if Type == "Mean":
            axis = st.selectbox("Select axis (Rows=0, Columns=1)", [0, 1])
            skipna = st.checkbox("Skip NA values?", True)
            numeric_only = st.checkbox("Include only numeric data?", False)
            return axis, skipna, numeric_only
        
        if Type == "Median":
            axis = st.selectbox("Select axis (Rows=0, Columns=1)", [0, 1])
            skipna = st.checkbox("Skip NA values?", True)
            numeric_only = st.checkbox("Include only numeric data?", False)
            return axis, skipna, numeric_only
        
        if Type == "Mode":
            axis = st.selectbox("Select axis (Rows=0, Columns=1)", [0, 1])
            numeric_only = st.checkbox("Include only numeric data?", False)
            dropna = st.checkbox("Drop missing values?", True)
            return axis, numeric_only, dropna

        if Type == "PctChange":
            periods = int(st.number_input("Number of periods to shift", min_value=1, value=1))
            return periods

        if Type == "Quantile":
            q = st.slider("Quantile to compute", min_value=0.0, max_value=1.0, value=0.5)
            axis = st.selectbox("Select axis (Columns=0, Rows=1)", [0, 1])
            numeric_only = st.checkbox("Include only numeric data?", False)
            interpolation = st.selectbox("Interpolation method", ['linear', 'lower', 'higher', 'midpoint', 'nearest'])
            return q, axis, numeric_only, interpolation

        if Type == "Rank":
            axis = st.selectbox("Select axis (Columns=0, Rows=1)", [0, 1])
            method = st.selectbox("Ranking method", ['average', 'min', 'max', 'first', 'dense'])
            numeric_only = st.checkbox("Include only numeric data?", False)
            na_option = st.selectbox("NA option", ['keep', 'top', 'bottom'])
            ascending = st.checkbox("Sort ascending?", True)
            pct = st.checkbox("Display rankings as percentage?", False)
            return axis, method, numeric_only, na_option, ascending, pct

        if Type == "SEM":
            axis = st.selectbox("Select axis (Columns=0, Rows=1)", [0, 1])
            skipna = st.checkbox("Skip NA values?", True)
            ddof = st.number_input("Degrees of freedom (ddof)", min_value=0, value=1)
            numeric_only = st.checkbox("Include only numeric data?", False)
            return axis, skipna, ddof, numeric_only

        if Type == "Skew":
            axis = st.selectbox("Select axis (Rows=0, Columns=1)", [0, 1])
            skipna = st.checkbox("Skip NA values?", True)
            numeric_only = st.checkbox("Include only numeric data?", False)
            return axis, skipna, numeric_only

        if Type == "STD":
            axis = st.selectbox("Select axis (columns=0, rows=1)", [0, 1])
            skipna = st.checkbox("Skip NA values?", True)
            ddof = st.number_input("Degrees of freedom (ddof)", min_value=0, value=1)
            numeric_only = st.checkbox("Include only numeric data?", False)
            return axis, skipna, ddof, numeric_only

        if Type == "Variance":
            axis = st.selectbox("Select axis (Rows=0, Columns=1)", [0, 1])
            skipna = st.checkbox("Skip NA values?", True)
            ddof = st.number_input("Degrees of freedom (ddof)", min_value=0, value=1)
            numeric_only = st.checkbox("Include only numeric data?", False)
            return axis, skipna, ddof, numeric_only

        if Type == "Covariance":
            min_periods = st.number_input("Minimum number of observations", min_value=0, value=1)
            ddof = st.number_input("Degrees of freedom (ddof)", min_value=0, value=1)
            numeric_only = st.checkbox("Include only numeric data?", False)
            return min_periods, ddof, numeric_only

        if Type == "Kurtosis":
            axis = st.selectbox("Select axis (Rows=0, Columns=1)", [0, 1])
            skipna = st.checkbox("Skip NA values?", True)
            numeric_only = st.checkbox("Include only numeric data?", False)
            return axis, skipna, numeric_only

        if Type == "Correlation":
            method = st.selectbox("Correlation method", ['pearson', 'kendall', 'spearman'])
            min_periods = st.number_input("Minimum number of observations per pair", min_value=1, value=1)
            numeric_only = st.checkbox("Include only numeric data?", False)
            return method, min_periods, numeric_only

        if Type == "Correlation With":
            other = st.file_uploader("Upload another dataset for correlation", type=['csv'])
            axis = st.selectbox("Select axis (Rows=0, Columns=1)", [0, 1])
            drop = st.checkbox("Drop missing values?", False)
            method = st.selectbox("Correlation method", ['pearson', 'kendall', 'spearman'])
            numeric_only = st.checkbox("Include only numeric data?", False)
            return other, axis, drop, method, numeric_only        
                
    
    def layout(self):
        st.sidebar.divider()
        st.sidebar.subheader("Apply Descriptive Statistics")
        if st.sidebar.checkbox("APPLY MEAN"):
            self.main("Mean")
        if st.sidebar.checkbox("APPLY MEDIAN"):
            self.main("Median")
        if st.sidebar.checkbox("APPLY MODE"):
            self.main("Mode")
        if st.sidebar.checkbox("APPLY CORRELATION"):
            self.main("Correlation")
        if st.sidebar.checkbox("APPLY COVARIANCE"):
            self.main("Covariance")
        if st.sidebar.checkbox("APPLY CORRELATION WITH"):
            self.main("Correlation With")
        if st.sidebar.checkbox("APPLY Skew"):
            self.main("Skew")
        if st.sidebar.checkbox("APPLY KURTOSIS"):
            self.main("Kurtosis")
        if st.sidebar.checkbox("APPLY STANDERED DEVIATION"):
            self.main("STD")
        if st.sidebar.checkbox("APPLY STANDERED ERROR OF THE MEAN"):
            self.main("SEM")
        if st.sidebar.checkbox("APPLY QUANTILE"):
            self.main("Quantile")
        if st.sidebar.checkbox("APPLY PERCENT CHANGE"):
            self.main("PctChange")
        if st.sidebar.checkbox("APPLY RANK"):
            self.main("Rank")
        

    def main(self, Type):
        col1, col2, col3 = st.columns([1, 2, 1])

        if Type == "Mean":
            with col1:
                st.subheader("Set parameters for calculating the mean")
                st.divider()
                mean_params = self.parameter_setup(Type)
                with col2:
                    if mean_params[0] == 0:  # Row-wise mean
                        rows = st.multiselect("Select rows for applying the mean", self.dataset.index)
                        columns = st.multiselect("Select columns for applying the mean", self.dataset.columns)
                        if rows and columns:
                            row_dataframe = self.dataset.loc[rows, columns]
                            st.subheader("YOUR DATAFRAME")
                            st.dataframe(row_dataframe)
                            st.subheader("APPLYING MEAN OVER SELECTED ROWS")
                            row_mean = row_dataframe.mean(axis=1, skipna=mean_params[1], numeric_only=mean_params[2])
                            st.dataframe(row_mean)
                    else:  # Column-wise mean
                        columns = st.multiselect("Select columns for applying the mean", self.dataset.columns)
                        if columns:
                            column_dataframe = self.dataset[columns]
                            st.subheader("YOUR DATAFRAME")
                            st.dataframe(column_dataframe)
                            st.subheader("APPLYING MEAN OVER SELECTED COLUMNS")
                            column_mean = column_dataframe.mean(axis=0, skipna=mean_params[1], numeric_only=mean_params[2])
                            st.dataframe(column_mean)

        if Type == "Median":
            with col1:
                st.subheader("Set parameters for calculating the median")
                st.divider()
                median_params = self.parameter_setup(Type)
                with col2:
                    if median_params[0] == 0:  # Row-wise median
                        rows = st.multiselect("Select rows for applying the median", self.dataset.index)
                        columns = st.multiselect("Select columns for applying the median", self.dataset.columns)
                        if rows and columns:
                            row_dataframe = self.dataset.loc[rows, columns]
                            st.subheader("YOUR DATAFRAME")
                            st.dataframe(row_dataframe)
                            st.subheader("APPLYING MEDIAN OVER SELECTED ROWS")
                            row_median = row_dataframe.median(axis=1, skipna=median_params[1], numeric_only=median_params[2])
                            st.dataframe(row_median)
                    else:  # Column-wise median
                        columns = st.multiselect("Select columns for applying the median", self.dataset.columns)
                        if columns:
                            column_dataframe = self.dataset[columns]
                            st.subheader("YOUR DATAFRAME")
                            st.dataframe(column_dataframe)
                            st.subheader("APPLYING MEDIAN OVER SELECTED COLUMNS")
                            column_median = column_dataframe.median(axis=0, skipna=median_params[1], numeric_only=median_params[2])
                            st.dataframe(column_median)

        if Type == "Mode":
            with col1:
                st.subheader("Set parameters for calculating the mode")
                st.divider()
                mode_params = self.parameter_setup(Type)
                with col2:
                    if mode_params[0] == 0:  # Row-wise mode
                        rows = st.multiselect("Select rows for applying the mode", self.dataset.index)
                        columns = st.multiselect("Select columns for applying the mode", self.dataset.columns)
                        if rows and columns:
                            row_dataframe = self.dataset.loc[rows, columns]
                            st.subheader("YOUR DATAFRAME")
                            st.dataframe(row_dataframe)
                            st.subheader("APPLYING MODE OVER SELECTED ROWS")
                            row_mode = row_dataframe.mode(axis=1, numeric_only=mode_params[1], dropna=mode_params[2])
                            st.dataframe(row_mode)
                    else:  # Column-wise mode
                        columns = st.multiselect("Select columns for applying the mode", self.dataset.columns)
                        if columns:
                            column_dataframe = self.dataset[columns]
                            st.subheader("YOUR DATAFRAME")
                            st.dataframe(column_dataframe)
                            st.subheader("APPLYING MODE OVER SELECTED COLUMNS")
                            column_mode = column_dataframe.mode(axis=0, numeric_only=mode_params[1], dropna=mode_params[2])
                            st.dataframe(column_mode)

        # Similar blocks can be added for other operations like Correlation, Covariance, Skew, Kurtosis, etc.

        if Type == "Correlation":
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Set parameters for calculating correlation")
                st.divider()
                cor_params = self.parameter_setup(Type)  # Corrected method call
                
            with col2:
                col4, col5 = st.columns([1, 1])
                
                with col4:
                    specific_index_columns = st.checkbox("Perform Operations Only On Selected Indexes And Columns")
                    
                with col5:
                    all_index_columns = st.checkbox("Perform Operations Only On Selected Columns With All Indexes")
                    
                # Perform correlation on selected columns with all indexes
                if all_index_columns:
                    selected_columns = st.multiselect("Select columns for correlation", self.dataset.columns)
                    if selected_columns:
                        corr_dataframe = self.dataset[selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(corr_dataframe)
                        
                        st.subheader("APPLYING CORRELATION")
                        correlation = corr_dataframe.corr(method=cor_params[0], 
                                                          min_periods=cor_params[1], 
                                                          numeric_only=cor_params[2])
                        st.dataframe(correlation)
                    else:
                        st.warning("Please select columns for correlation.")
                
                # Perform correlation on selected indexes and columns
                if specific_index_columns:
                    selected_indexes = st.multiselect("Select the indexes that you want", self.dataset.index)
                    selected_columns = st.multiselect("Select columns for correlation", self.dataset.columns)
                    
                    if selected_indexes and selected_columns:
                        # Subset the DataFrame using the selected indexes and columns
                        corr_dataframe = self.dataset.loc[selected_indexes, selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(corr_dataframe)
                        
                        st.subheader("APPLYING CORRELATION")
                        correlation = corr_dataframe.corr(method=cor_params[0], 
                                                          min_periods=cor_params[1], 
                                                          numeric_only=cor_params[2])
                        st.dataframe(correlation)
                    else:
                        st.warning("Please select both indexes and columns for correlation.")
        if Type == "Correlation With":
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Set parameters for calculating correlation With")
                st.divider()
                cor_params = self.parameter_setup(Type)  # Correct method call
                
            with col2:
                col4, col5 = st.columns([1, 1])
                
                with col4:
                    specific_index_columns = st.checkbox("Perform Operations Only On Selected Indexes And Columns")
                    
                with col5:
                    all_index_columns = st.checkbox("Perform Operations Only On Selected Columns With All Indexes")
                
                # Ensure a secondary dataset is uploaded for "Correlation With"
                if cor_params[0] is not None:
                    other_dataset = pd.read_csv(cor_params[0])

                    # Perform correlation on selected columns with all indexes
                    if all_index_columns:
                        selected_columns = st.multiselect("Select columns for correlation from main dataset", self.dataset.columns)
                        other_selected_columns = st.multiselect("Select columns for correlation from uploaded dataset", other_dataset.columns)
                        
                        if selected_columns and other_selected_columns:
                            corr_dataframe_main = self.dataset[selected_columns]
                            corr_dataframe_other = other_dataset[other_selected_columns]
                            
                            st.subheader("Main Dataset")
                            st.dataframe(corr_dataframe_main)
                            st.subheader("Uploaded Dataset")
                            st.dataframe(corr_dataframe_other)
                            
                            st.subheader("APPLYING CORRELATION")
                            # Perform correlation between main and other datasets (column-wise)
                            correlation = corr_dataframe_main.corrwith(corr_dataframe_other, axis=cor_params[1], drop=cor_params[2], method=cor_params[3], numeric_only=cor_params[4])
                            st.dataframe(correlation)
                        else:
                            st.warning("Please select columns from both datasets for correlation.")
                    
                    # Perform correlation on selected indexes and columns
                    if specific_index_columns:
                        selected_indexes = st.multiselect("Select indexes from main dataset", self.dataset.index)
                        selected_columns = st.multiselect("Select columns from main dataset", self.dataset.columns)
                        other_selected_columns = st.multiselect("Select columns from uploaded dataset", other_dataset.columns)
                        
                        if selected_indexes and selected_columns and other_selected_columns:
                            # Subset the datasets using the selected indexes and columns
                            corr_dataframe_main = self.dataset.loc[selected_indexes, selected_columns]
                            corr_dataframe_other = other_dataset[other_selected_columns]
                            
                            st.subheader("Main Dataset")
                            st.dataframe(corr_dataframe_main)
                            st.subheader("Uploaded Dataset")
                            st.dataframe(corr_dataframe_other)
                            
                            st.subheader("APPLYING CORRELATION")
                            # Perform correlation between main and other datasets (column-wise)
                            correlation = corr_dataframe_main.corrwith(corr_dataframe_other, axis=cor_params[1], drop=cor_params[2], method=cor_params[3], numeric_only=cor_params[4])
                            st.dataframe(correlation)
                        else:
                            st.warning("Please select indexes and columns from both datasets for correlation.")
                else:
                    st.warning("Please upload a dataset for correlation.")
                        

        if Type == "Covariance":
            with col1:
                st.subheader("Set parameters for calculating covariance")
                st.divider()

                # Collect common parameters for covariance calculation
                cov_params = self.parameter_setup(Type)

            with col2:
                col4, col5 = st.columns([1, 1])

                with col4:
                    specific_index_columns = st.checkbox("Perform Operations Only On Selected Indexes And Columns")

                with col5:
                    all_index_columns = st.checkbox("Perform Operations Only On Selected Columns With All Indexes")

                if all_index_columns:
                    # Select columns for covariance
                    selected_columns = st.multiselect("Select columns for covariance", self.dataset.columns)

                    if selected_columns:  # Ensure that columns are selected
                        cov_dataframe = self.dataset[selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(cov_dataframe)

                        st.subheader("APPLYING COVARIANCE")
                        covariance = cov_dataframe.cov(cov_params[0],cov_params[1],cov_params[2])
                        st.dataframe(covariance)
                    else:
                        st.warning("Please select columns for covariance.")

                if specific_index_columns:
                    # Select indexes and columns for covariance
                    selected_indexes = st.multiselect("Select the indexes that you want", self.dataset.index)
                    selected_columns = st.multiselect("Select columns for covariance", self.dataset.columns)

                    if selected_indexes and selected_columns:  # Ensure that both indexes and columns are selected
                        cov_dataframe = self.dataset.loc[selected_indexes, selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(cov_dataframe)

                        st.subheader("APPLYING COVARIANCE")
                        covariance = cov_dataframe.cov(cov_params[0],cov_params[1],cov_params[2])
                        st.dataframe(covariance)
                    else:
                        st.warning("Please select both indexes and columns for covariance.")
        if Type == "Kurtosis":
            with col1:
                st.subheader("Set parameters for calculating kurtosis")
                st.divider()
                kurtosis_params = self.parameter_setup(Type)

            with col2:
                if kurtosis_params[0] == 0:  
                    rows = st.multiselect("Select rows for applying kurtosis", self.dataset.index)
                    selected_columns = st.multiselect("Select columns for kurtosis", self.dataset.columns)

                    if rows and selected_columns:
                        row_dataframe = self.dataset.loc[rows, selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(row_dataframe)

                        st.subheader("APPLYING KURTOSIS OVER SELECTED ROWS")
                        row_kurtosis = row_dataframe.kurtosis(
                            axis=1, skipna=kurtosis_params[1], numeric_only=kurtosis_params[2]
                        )
                        st.dataframe(row_kurtosis)
                    else:
                        st.warning("Please select both rows and columns for applying kurtosis.")
                
                else:  
                    columns = st.multiselect("Select columns for applying kurtosis", self.dataset.columns)

                    if columns:
                        column_dataframe = self.dataset[columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(column_dataframe)

                        st.subheader("APPLYING KURTOSIS OVER SELECTED COLUMNS")
                        column_kurtosis = column_dataframe.kurtosis(
                            axis=0, skipna=kurtosis_params[1], numeric_only=kurtosis_params[2]
                        )
                        st.dataframe(column_kurtosis)
                    else:
                        st.warning("Please select columns for applying kurtosis.")

        if Type == "PctChange":
            with col1:
                st.subheader("Set parameters for calculating Percent Change")
                st.divider()

                pct_params = self.parameter_setup(Type)

            with col2:
                col4, col5 = st.columns([1, 1])
                with col4:
                    specific_index_columns = st.checkbox("Perform Operations Only On Selected Indexes And Columns")
                with col5:
                    all_index_columns = st.checkbox("Perform Operations Only On Selected Columns With All Indexes")

                # Case 1: Apply pct_change to selected columns for all indexes
                if all_index_columns:
                    selected_columns = st.multiselect("Select columns for pct_change", self.dataset.columns)
                    if selected_columns:
                        pct_dataframe = self.dataset[selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(pct_dataframe)
                        st.subheader("APPLYING Percent Change Over Selected Columns")
                        pct_change = pct_dataframe.pct_change()
                        st.dataframe(pct_change)
                    else:
                        st.warning("Please select columns for applying Percent Change.")

                if specific_index_columns:
                    selected_indexes = st.multiselect("Select indexes for applying Percent Change", self.dataset.index)
                    selected_columns = st.multiselect("Select columns for pct_change", self.dataset.columns)

                    if selected_indexes and selected_columns:
                        pct_dataframe = self.dataset.loc[selected_indexes, selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(pct_dataframe)
                        st.subheader("APPLYING Percent Change Over Selected Indexes and Columns")
                        pct_change = pct_dataframe.pct_change()
                        st.dataframe(pct_change)
                    else:
                        st.warning("Please select both indexes and columns for applying Percent Change.")

        if Type=="Quantile":
            with col1:
                st.subheader("Set parameters for calculating quantile")
                st.divider()
                quantile_params = self.parameter_setup(Type)
                with col2:
                    col4,col5=st.columns([1,1])
                    with col4:
                        specific_index_columns=st.checkbox("Perform Operations Only On Selected Indexes And Columns")
                    with col5:
                        All_index_columns=st.checkbox("Perform Operations Only On Selected Columns With All Indexes")
                    if All_index_columns:
                        selected_columns = st.multiselect("Select columns for pct_change", self.dataset.columns)
                        quantile_dataframe = self.dataset[selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(quantile_dataframe)
                        st.subheader("APPLYING Percent Change")
                        quantile_df = quantile_dataframe.quantile(quantile_params[0],quantile_params[1],quantile_params[2],quantile_params[3])
                        st.dataframe(quantile_df)
                    if specific_index_columns:
                        selected_indexes=st.multiselect("select the indexes that you want",self.dataset.index)
                        selected_columns = st.multiselect("Select columns for correlation", self.dataset.columns)
                        quantile_dataframe = self.dataset.loc[selected_indexes,selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(quantile_dataframe)
                        st.subheader("APPLYING Percent Change")
                        quantile_df = quantile_dataframe.quantile(quantile_params[0],quantile_params[1],quantile_params[2],quantile_params[3])
                        st.dataframe(quantile_df)
        if Type=="Rank":
            with col1:
                st.subheader("Set parameters for calculating Rank")
                st.divider()
                rank_params = self.parameter_setup(Type)
                with col2:
                    col4,col5=st.columns([1,1])
                    with col4:
                        specific_index_columns=st.checkbox("Perform Operations Only On Selected Indexes And Columns")
                    with col5:
                        All_index_columns=st.checkbox("Perform Operations Only On Selected Columns With All Indexes")
                    if All_index_columns:
                        selected_columns = st.multiselect("Select columns for rank", self.dataset.columns)
                        rank_dataframe = self.dataset[selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(rank_dataframe)
                        st.subheader("APPLYING RANK")
                        rank_df = rank_dataframe.rank(rank_params[0],rank_params[1],rank_params[2],rank_params[3],rank_params[4],rank_params[5])
                        st.dataframe(rank_df)
                    if specific_index_columns:
                        selected_indexes=st.multiselect("select the indexes that you want",self.dataset.index)
                        selected_columns = st.multiselect("Select columns for correlation", self.dataset.columns)
                        rank_dataframe = self.dataset.loc[selected_indexes,selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(rank_dataframe)
                        st.subheader("APPLYING RANK")
                        rank_df = rank_dataframe.rank(rank_params[0],rank_params[1],rank_params[2],rank_params[3],rank_params[4],rank_params[5])
                        st.dataframe(rank_df)
        if Type == "SEM":
            with col1:
                st.subheader("Set parameters for calculating standered error of the mean")
                st.divider()
                sem_params = self.parameter_setup(Type)
                with col2:
                    col4,col5=st.columns([1,1])
                    with col4:
                        specific_index_columns=st.checkbox("Perform Operations Only On Selected Indexes And Columns")
                    with col5:
                        All_index_columns=st.checkbox("Perform Operations Only On Selected Columns With All Indexes")
                    if All_index_columns:
                        selected_columns = st.multiselect("Select columns for standered error of the mean", self.dataset.columns)
                        sem_dataframe = self.dataset[selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(sem_dataframe)
                        st.subheader("APPLYING STANDERED ERROR OF THE MEAN")
                        sem_df = sem_dataframe.sem(sem_params[0],sem_params[1],sem_params[2],sem_params[3])
                        st.dataframe(sem_df)
                    if specific_index_columns:
                        selected_indexes=st.multiselect("select the indexes that you want",self.dataset.index)
                        selected_columns = st.multiselect("Select columns for correlation", self.dataset.columns)
                        sem_dataframe = self.dataset.loc[selected_indexes,selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(sem_dataframe)
                        st.subheader("APPLYING STANDERED ERROR OF THE MEAN")
                        sem_df = sem_dataframe.sem(sem_params[0],sem_params[1],sem_params[2],sem_params[3])
                        st.dataframe(sem_df)
        if Type == "STD":
            with col1:
                st.subheader("Set parameters for calculating standered deviation")
                st.divider()
                std_params = self.parameter_setup(Type)
                with col2:
                    col4,col5=st.columns([1,1])
                    with col4:
                        specific_index_columns=st.checkbox("Perform Operations Only On Selected Indexes And Columns")
                    with col5:
                        All_index_columns=st.checkbox("Perform Operations Only On Selected Columns With All Indexes")
                    if All_index_columns:
                        selected_columns = st.multiselect("Select columns for standered error of the mean", self.dataset.columns)
                        std_dataframe = self.dataset[selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(std_dataframe)
                        st.subheader("APPLYING STANDERED DEVIATION")
                        std_df = std_dataframe.std(std_params[0],std_params[1],std_params[2],std_params[3])
                        st.dataframe(std_df)
                    if specific_index_columns:
                        selected_indexes=st.multiselect("select the indexes that you want",self.dataset.index)
                        selected_columns = st.multiselect("Select columns for correlation", self.dataset.columns)
                        std_dataframe = self.dataset.loc[selected_indexes,selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(std_dataframe)
                        st.subheader("APPLYING STANDERED DEVIATION")
                        std_df = std_dataframe.std(std_params[0],std_params[1],std_params[2],std_params[3])
                        st.dataframe(std_df)
        if Type == "VAR":
            with col1:
                st.subheader("Set parameters for calculating varience")
                st.divider()
                var_params = self.parameter_setup(Type)
                with col2:
                    col4,col5=st.columns([1,1])
                    with col4:
                        specific_index_columns=st.checkbox("Perform Operations Only On Selected Indexes And Columns")
                    with col5:
                        All_index_columns=st.checkbox("Perform Operations Only On Selected Columns With All Indexes")
                    if All_index_columns:
                        selected_columns = st.multiselect("Select columns for standered error of the mean", self.dataset.columns)
                        var_dataframe = self.dataset[selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(var_dataframe)
                        st.subheader("APPLYING VARIENCE")
                        var_df = var_dataframe.var(var_params[0],var_params[1],var_params[2],var_params[3])
                        st.dataframe(var_df)
                    if specific_index_columns:
                        selected_indexes=st.multiselect("select the indexes that you want",self.dataset.index)
                        selected_columns = st.multiselect("Select columns for correlation", self.dataset.columns)
                        var_dataframe = self.dataset[selected_columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(var_dataframe)
                        st.subheader("APPLYING VARIENCE")
                        var_df = var_dataframe.var(var_params[0],var_params[1],var_params[2],var_params[3])
                        st.dataframe(var_df)
        if Type == "Skew":
            with col1:
                st.subheader("Set parameters for calculating the mean")
                st.divider()
                skew_params = self.parameter_setup(Type)
                with col2:
                    if skew_params[0] == 0:
                        rows = st.multiselect("Select rows for applying the skew", self.dataset.index)
                        columns = st.multiselect("Select columns for applying the skew", self.dataset.columns)
                        row_dataframe = self.dataset.loc[rows,columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(row_dataframe)
                        st.subheader("APPLYING SKEW OVER SELECTED ROWS")
                        row_skew = row_dataframe.skew(1,skew_params[1], skew_params[2])
                        st.dataframe(row_skew)
                    else:
                        columns = st.multiselect("Select columns for applying the skew", self.dataset.columns)
                        column_dataframe = self.dataset[columns]
                        st.subheader("YOUR DATAFRAME")
                        st.dataframe(column_dataframe)
                        st.subheader("APPLYING SKEW OVER SELECTED COLUMNS")
                        column_skew = column_dataframe.skew(0,skew_params[1], skew_params[2])
                        st.dataframe(column_skew)

class Trigonometry:
    def __init__(self, dataset):
        self.dataset = dataset

    def equalityFunction(self, columns):
        if len(columns) % 2 == 0:
            odd_list = []
            even_list = []
            for i in range(len(columns)):
                if i % 2 == 0:
                    even_list.append(columns[i])
                else:
                    odd_list.append(columns[i])
            return odd_list, even_list
        else:
            st.text("You have an unequal number of pairs to compute pairs of hypotenuse. Please remove one column from the list or select one more column to compute hypotenuses.")
            return [], []

    def equalityPerform(self, odd_list, even_list, new_dataframe):
        for i in range(len(odd_list)):
            new_dataframe[f"HYPOT({odd_list[i]},{even_list[i]})"] = np.hypot(
                new_dataframe[odd_list[i]], new_dataframe[even_list[i]]
            )
            st.info(f"SUCCESS FOR CREATING HYPOTENUSE VALUES FOR {odd_list[i]},{even_list[i]}")
            st.dataframe(new_dataframe[[f"HYPOT({odd_list[i]},{even_list[i]})"]])

    def layout(self):
        st.sidebar.info("TRIGONOMETRIC FUNCTIONS")
        trigonometric_functions = [
            "SINE", "COSINE", "TANGENT", "COTANGENT", "SECANT", "COSECANT",
            "SINE INVERSE", "COSINE INVERSE", "TANGENT INVERSE", "COTANGENT INVERSE",
            "SECANT INVERSE", "COSECANT INVERSE", "HYPOTENUSE", "DEGREES", "RADIANS"
        ]
        for function in trigonometric_functions:
            if st.sidebar.checkbox(f"APPLY {function}"):
                self.main(function)

    def main(self, Type):
        tab1, tab2 = st.tabs(["Perform Operations here", "Check Your Results Here"])
        with tab1:
            select_columns = st.multiselect(f"Select the columns to apply {Type}", self.dataset.columns)
            if select_columns:
                new_dataFrame = self.dataset[select_columns].copy()
                st.success("Your Selected Data Frame")
                st.dataframe(new_dataFrame)

                for i in range(len(select_columns)):
                    # Get the column name and its values
                    column_name = select_columns[i]
                    select_col = new_dataFrame[column_name].values

                    if Type != "HYPOTENUSE":
                        self.apply_trigonometric_function(new_dataFrame, select_col, column_name, Type)

                    if Type == "HYPOTENUSE":
                        odd_list, even_list = self.equalityFunction(select_columns)
                        if odd_list and even_list:
                            if st.checkbox("CONFIRM THE ABOVE COLUMNS TO APPLY THE HYPOTENUSE"):
                                self.equalityPerform(odd_list, even_list, new_dataFrame)
                            else:
                                st.warning("PLEASE CONFIRM FIRST TO CALCULATE THE PAIRS OF HYPOTENUSE")

                    st.success(f"After applying the {Type} function to {column_name}")
                    st.dataframe(new_dataFrame[[column_name, f"{column_name}-{Type}"]])
                    with tab2:
                        st.success("YOUR OVERALL DATA WITH THE RESULTS OF THE TRIGNOMETRIC FUNCTIONS")
                        st.dataframe(new_dataFrame)
            else:
                st.info(f"PLEASE SELECT ONE OR MORE COLUMNS TO {Type}")

    def apply_trigonometric_function(self, new_dataFrame, select_col, column_name, Type):
        if Type == "SINE":
            new_dataFrame[f"{column_name}-{Type}"] = np.sin(np.deg2rad(select_col))
        elif Type == "COSINE":
            new_dataFrame[f"{column_name}-{Type}"] = np.cos(np.deg2rad(select_col))
        elif Type == "TANGENT":
            new_dataFrame[f"{column_name}-{Type}"] = np.tan(np.deg2rad(select_col))
        elif Type == "COTANGENT":
            new_dataFrame[f"{column_name}-{Type}"] = 1 / np.tan(np.deg2rad(select_col))
        elif Type == "SECANT":
            new_dataFrame[f"{column_name}-{Type}"] = 1 / np.cos(np.deg2rad(select_col))
        elif Type == "COSECANT":
            new_dataFrame[f"{column_name}-{Type}"] = 1 / np.sin(np.deg2rad(select_col))
        elif Type == "SINE INVERSE":
            new_dataFrame[f"{column_name}-{Type}"] = np.arcsin(np.deg2rad(select_col))
        elif Type == "COSINE INVERSE":
            new_dataFrame[f"{column_name}-{Type}"] = np.arccos(np.deg2rad(select_col))
        elif Type == "TANGENT INVERSE":
            new_dataFrame[f"{column_name}-{Type}"] = np.arctan(np.deg2rad(select_col))
        elif Type == "COTANGENT INVERSE":
            new_dataFrame[f"{column_name}-{Type}"] = np.arccot(np.deg2rad(select_col))
        elif Type == "SECANT INVERSE":
            new_dataFrame[f"{column_name}-{Type}"] = np.arccos(1 / np.deg2rad(select_col))
        elif Type == "COSECANT INVERSE":
            new_dataFrame[f"{column_name}-{Type}"] = np.arcsin(1 / np.deg2rad(select_col))
        elif Type == "DEGREES":
            new_dataFrame[f"{column_name}-{Type}"] = np.rad2deg(np.deg2rad(select_col))
        elif Type == "RADIANS":
            new_dataFrame[f"{column_name}-{Type}"] = np.deg2rad(np.deg2rad(select_col))
        else:
            st.warning(f"Unknown operation type: {Type}")

class BasicMathematicalOperations:
    def __init__(self, dataset):
        self.dataset = dataset

    def _layout_(self):
        st.sidebar.info("BASIC MATHEMATICAL OPERATIONS")
        
        add = st.sidebar.checkbox("Add a Scalar Value to the Dataset")
        if add:
            self.main("ADD")
        sub = st.sidebar.checkbox("Element-wise Subtraction with a Scalar")
        if sub:
            self.main("SUB")
        mul = st.sidebar.checkbox("Element-wise Multiplication by a Scalar")
        if mul:
            self.main("MUL")
        div = st.sidebar.checkbox("Element-wise Division by a Scalar")
        if div:
            self.main("DIV")
        floor_div = st.sidebar.checkbox("Element-wise Floor Division by a Scalar")
        if floor_div:
            self.main("FLOOR DIV")
        mod = st.sidebar.checkbox("Element-wise Modulus with a Scalar")
        if mod:
            self.main("MOD")
        power = st.sidebar.checkbox("Element-wise Power with a Scalar")
        if power:
            self.main("POW")
        


    def main(self,Type):
        tab1,tab2=st.tabs(["Basic","Advanced"])
        with tab1:
            col1, col2 = st.columns(2)

            # Column 1: Perform operations on the entire dataset (all rows or all columns)
            with col1:
                st.write("Select the axis to perform operations on the entire dataset:")
                axis_option_all = st.radio(
                    "Perform operations on:",
                    ('Columns', 'Rows'),
                    key="axis_all"
                )

                # Axis for applying the operations: 1 for columns, 0 for rows
                axis_all = 1 if axis_option_all == 'Columns' else 0
                SelectColumns=st.multiselect("Select the columns that you want",self.dataset.columns)

                # Apply operations on the entire dataset
                if Type=="ADD":
                    scalar = st.number_input("Enter a scalar value to add:", value=0.0)
                    result = self.dataset[SelectColumns].add(scalar, axis=axis_all)
                    st.write("Result after Addition (on entire dataset):")
                    st.dataframe(result)

                if Type=="SUB":
                    scalar = st.number_input("Enter a scalar value to subtract:", value=0.0)
                    result = self.dataset[SelectColumns].sub(scalar, axis=axis_all)
                    st.write("Result after Subtraction (on entire dataset):")
                    st.dataframe(result)

                if Type=="MUL":
                    scalar = st.number_input("Enter a scalar value to multiply:", value=1.0)
                    result = self.dataset[SelectColumns].mul(scalar, axis=axis_all)
                    st.write("Result after Multiplication (on entire dataset):")
                    st.dataframe(result)

                if Type=="DIV":
                    scalar = st.number_input("Enter a scalar value to divide by:", value=1.0)
                    if scalar != 0:
                        result = self.dataset[SelectColumns].truediv(scalar, axis=axis_all)
                        st.write("Result after Division (on entire dataset):")
                        st.dataframe(result)
                    else:
                        st.error("Cannot divide by zero!")

                if Type=="FLOOR DIV":
                    scalar = st.number_input("Enter a scalar value for floor division:", value=1.0)
                    if scalar != 0:
                        result = self.dataset[SelectColumns].floordiv(scalar, axis=axis_all)
                        st.write("Result after Floor Division (on entire dataset):")
                        st.dataframe(result)
                    else:
                        st.error("Cannot floor divide by zero!")

                if Type=="MOD":
                    scalar = st.number_input("Enter a scalar value for modulus:", value=1.0)
                    if scalar != 0:
                        result = self.dataset[SelectColumns].mod(scalar, axis=axis_all)
                        st.write("Result after Modulus Operation (on entire dataset):")
                        st.dataframe(result)
                    else:
                        st.error("Cannot perform modulus with zero!")

                if Type=="POW":
                    scalar = st.number_input("Enter a scalar value for exponentiation:", value=1.0)
                    result = self.dataset[SelectColumns].pow(scalar, axis=axis_all)
                    st.write("Result after Exponentiation (on entire dataset):")
                    st.dataframe(result)

            # Column 2: Perform operations on a subset of the dataset
            with col2:
                st.write("Subset the data:")
                selected_columns = st.multiselect("Select columns", self.dataset.columns)
                selected_indices = st.multiselect("Select rows", self.dataset.index)

                # Get the subset of the data
                if selected_columns and selected_indices:
                    subset = self.dataset.loc[selected_indices, selected_columns]
                    st.write("Selected Subset:")
                    st.dataframe(subset)
                    
                    # Select axis for subset operation
                    axis_option_subset = st.radio(
                        "Perform operations on:",
                        ('Columns', 'Rows'),
                        key="axis_subset"
                    )

                    # Axis for applying the operations: 1 for columns, 0 for rows
                    axis_subset = 1 if axis_option_subset == 'Columns' else 0
                    # Apply operations on the subset of data
                    if Type=="ADD":
                        scalar = st.number_input("Enter a scalar value to add (subset):", value=0.0, key="scalar_add_subset")
                        result = subset.add(scalar, axis=axis_subset)
                        st.write("Result after Addition (on subset):")
                        st.dataframe(result)

                    if Type=="SUB":
                        scalar = st.number_input("Enter a scalar value to subtract (subset):", value=0.0, key="scalar_subtract_subset")
                        result = subset.subtract(scalar, axis=axis_subset)
                        st.write("Result after Subtraction (on subset):")
                        st.dataframe(result)

                    if Type=="MUL":
                        scalar = st.number_input("Enter a scalar value to multiply (subset):", value=1.0, key="scalar_multiply_subset")
                        result = subset.multiply(scalar, axis=axis_subset)
                        st.write("Result after Multiplication (on subset):")
                        st.dataframe(result)

                    if Type=="DIV":
                        scalar = st.number_input("Enter a scalar value to divide by (subset):", value=1.0, key="scalar_divide_subset")
                        if scalar != 0:
                            result = subset.divide(scalar, axis=axis_subset)
                            st.write("Result after Division (on subset):")
                            st.dataframe(result)
                        else:
                            st.error("Cannot divide by zero!")

                    if Type=="FLOOR DIV":
                        scalar = st.number_input("Enter a scalar value for floor division (subset):", value=1.0, key="scalar_floor_divide_subset")
                        if scalar != 0:
                            result = subset.floordiv(scalar, axis=axis_subset)
                            st.write("Result after Floor Division (on subset):")
                            st.dataframe(result)
                        else:
                            st.error("Cannot floor divide by zero!")

                    if Type=="MOD":
                        scalar = st.number_input("Enter a scalar value for modulus (subset):", value=1.0, key="scalar_modulus_subset")
                        if scalar != 0:
                            result = subset.mod(scalar, axis=axis_subset)
                            st.write("Result after Modulus Operation (on subset):")
                            st.dataframe(result)
                        else:
                            st.error("Cannot perform modulus with zero!")

                    if Type=="POW":
                        scalar = st.number_input("Enter a scalar value for exponentiation (subset):", value=1.0, key="scalar_power_subset")
                        result = subset.pow(scalar, axis=axis_subset)
                        st.write("Result after Exponentiation (on subset):")
                        st.dataframe(result)

                else:
                    st.write("Please select both rows and columns to create a subset.")
       
                    
class Financial_Functions:
    def __init__(self,dataset):
        self.dataset=dataset
    def layout(self):
        st.sidebar.info("FINANCIAL FUNCTIONS")
        fv=st.sidebar.checkbox("Find Future Value")
        if fv:
            self.main("fv")
        pv=st.sidebar.checkbox("Find Present Value")
        if pv:
            self.main("pv")
        npv=st.sidebar.checkbox("Find Net Present Value")
        if npv:
            self.main("npv")
        pmt=st.sidebar.checkbox(" Compute the payment against loan principal plus interest")
        if pmt:
            self.main("pmt")
        ppmt=st.sidebar.checkbox(" Compute the payment against loan principal.")
        if ppmt:
            self.main("ppmt")
        ipmt=st.sidebar.checkbox("Compute the interest portion of a payment.")
        if ipmt:
            self.main("ipmt")
        irr=st.sidebar.checkbox(" Return the Internal Rate of Return (IRR)")
        if irr:
            self.main("irr")
        mirr=st.sidebar.checkbox(" Modified internal rate of return.")
        if mirr:
            self.main("mirr")
        nper=st.sidebar.checkbox("Compute the number of periodic payments.")
        if nper:
            self.main("nper")
        rate=st.sidebar.checkbox("Compute the rate of interest per period.")
        if rate:
            self.main("rate")
    def work_with_datasets(self, Type, dataset):
        resultant_dataframe = pd.DataFrame()
        
        if Type == "fv":
            st.success("Provide dataset for Future Value calculation")
            
            rate_col = st.selectbox("Select column for 'RATE'", self.dataset.columns)
            nper_col = st.selectbox("Select column for 'NPER'", self.dataset.columns)
            pmt_col = st.selectbox("Select column for 'PMT'", self.dataset.columns)
            pv_col = st.selectbox("Select column for 'PV'", self.dataset.columns)
            
            # Retrieve values
            resultant_dataframe[rate_col] = self.dataset[rate_col]
            resultant_dataframe[nper_col] = self.dataset[nper_col]
            resultant_dataframe[pmt_col] = self.dataset[pmt_col]
            resultant_dataframe[pv_col] = self.dataset[pv_col]
            
            if rate_col and nper_col and pmt_col and pv_col:
                rate = np.array(self.dataset[rate_col].values)
                nper = np.array(self.dataset[nper_col].values)
                pmt = np.array(self.dataset[pmt_col].values)
                pv = np.array(self.dataset[pv_col].values)

                st.info("Successfully computed")
                # Calculate future value
                resultant_dataframe["Future Value"] = npf.fv(rate, nper, pmt, pv)
                st.dataframe(resultant_dataframe)

        elif Type == "pv":
            st.success("Provide dataset for Present Value calculation")
            
            rate_col = st.selectbox("Select column for 'RATE'", self.dataset.columns)
            nper_col = st.selectbox("Select column for 'NPER'", self.dataset.columns)
            pmt_col = st.selectbox("Select column for 'PMT'", self.dataset.columns)
            fv_col = st.selectbox("Select column for 'FV' (optional)", self.dataset.columns)
            
            # Retrieve values
            resultant_dataframe[rate_col] = self.dataset[rate_col]
            resultant_dataframe[nper_col] = self.dataset[nper_col]
            resultant_dataframe[pmt_col] = self.dataset[pmt_col]
            resultant_dataframe[fv_col] = self.dataset[fv_col]
            
            if rate_col and nper_col and pmt_col:
                rate = np.array(self.dataset[rate_col].values)
                nper = np.array(self.dataset[nper_col].values)
                pmt = np.array(self.dataset[pmt_col].values)
                fv = np.array(self.dataset[fv_col].values) if fv_col else 0

                st.info("Successfully computed")
                # Calculate present value
                resultant_dataframe["Present Value"] = npf.pv(rate, nper, pmt, fv)
                st.dataframe(resultant_dataframe)

        elif Type == "npv":
            st.success("Provide dataset for Net Present Value calculation")

            rate_col = st.selectbox("Select column for 'RATE'", self.dataset.columns)
            values_col = st.multiselect("Select the column(s) for cash flow values", self.dataset.columns)
            
            resultant_dataframe[rate_col] = self.dataset[rate_col]
            for col in values_col:
                resultant_dataframe[col] = self.dataset[col]
            
            if rate_col and values_col:
                rate = self.dataset[rate_col].values[0]  # Assuming a single rate value for NPV calculation
                values = self.dataset[values_col].sum(axis=1)  # Summing across selected columns for each row

                st.info("Successfully computed")
                # Calculate Net Present Value
                resultant_dataframe["Net Present Value"] = npf.npv(rate, values)
                st.dataframe(resultant_dataframe)

        elif Type == "pmt":
            st.success("Provide dataset for Payment calculation")

            rate_col = st.selectbox("Select column for 'RATE'", self.dataset.columns)
            nper_col = st.selectbox("Select column for 'NPER'", self.dataset.columns)
            pv_col = st.selectbox("Select column for 'PV'", self.dataset.columns)
            fv_col = st.selectbox("Select column for 'FV' (optional)", self.dataset.columns)

            resultant_dataframe[rate_col] = self.dataset[rate_col]
            resultant_dataframe[nper_col] = self.dataset[nper_col]
            resultant_dataframe[pv_col] = self.dataset[pv_col]
            resultant_dataframe[fv_col] = self.dataset[fv_col]

            if rate_col and nper_col and pv_col:
                rate = np.array(self.dataset[rate_col].values)
                nper = np.array(self.dataset[nper_col].values)
                pv = np.array(self.dataset[pv_col].values)
                fv = np.array(self.dataset[fv_col].values) if fv_col else 0

                st.info("Successfully computed")
                # Calculate payment
                resultant_dataframe["Payment"] = npf.pmt(rate, nper, pv, fv)
                st.dataframe(resultant_dataframe)

        elif Type == "ppmt":
            st.success("Provide dataset for Principal Payment calculation")

            rate_col = st.selectbox("Select column for 'RATE'", self.dataset.columns)
            per_col = st.selectbox("Select column for 'PER'", self.dataset.columns)
            nper_col = st.selectbox("Select column for 'NPER'", self.dataset.columns)
            pv_col = st.selectbox("Select column for 'PV'", self.dataset.columns)
            fv_col = st.selectbox("Select column for 'FV' (optional)", self.dataset.columns)

            resultant_dataframe[rate_col] = self.dataset[rate_col]
            resultant_dataframe[nper_col] = self.dataset[nper_col]
            resultant_dataframe[pv_col] = self.dataset[pv_col]
            resultant_dataframe[fv_col] = self.dataset[fv_col]
            resultant_dataframe[per_col] = self.dataset[per_col]

            if rate_col and per_col and nper_col and pv_col:
                rate = np.array(self.dataset[rate_col].values)
                per = np.array(self.dataset[per_col].values)
                nper = np.array(self.dataset[nper_col].values)
                pv = np.array(self.dataset[pv_col].values)
                fv = np.array(self.dataset[fv_col].values) if fv_col else 0

                st.info("Successfully computed")
                # Calculate principal payment
                resultant_dataframe["Principal Payment"] = npf.ppmt(rate, per, nper, pv, fv)
                st.dataframe(resultant_dataframe)

        elif Type == "ipmt":
            st.success("Provide dataset for Interest Payment calculation")

            rate_col = st.selectbox("Select column for 'RATE'", self.dataset.columns)
            per_col = st.selectbox("Select column for 'PER'", self.dataset.columns)
            nper_col = st.selectbox("Select column for 'NPER'", self.dataset.columns)
            pv_col = st.selectbox("Select column for 'PV'", self.dataset.columns)
            fv_col = st.selectbox("Select column for 'FV' (optional)", self.dataset.columns)

            resultant_dataframe[rate_col] = self.dataset[rate_col]
            resultant_dataframe[nper_col] = self.dataset[nper_col]
            resultant_dataframe[pv_col] = self.dataset[pv_col]
            resultant_dataframe[fv_col] = self.dataset[fv_col]
            resultant_dataframe[per_col] = self.dataset[per_col]

            if rate_col and per_col and nper_col and pv_col:
                rate = np.array(self.dataset[rate_col].values)
                per = np.array(self.dataset[per_col].values)
                nper = np.array(self.dataset[nper_col].values)
                pv = np.array(self.dataset[pv_col].values)
                fv = np.array(self.dataset[fv_col].values) if fv_col else 0

                st.info("Successfully computed")
                # Calculate interest payment
                resultant_dataframe["Interest Payment"] = npf.ipmt(rate, per, nper, pv, fv)
                st.dataframe(resultant_dataframe)

        elif Type == "irr":
            st.success("Provide dataset for IRR calculation")

            values_col = st.multiselect("Select the column(s) for cash flow values", self.dataset.columns)
            resultant_dataframe[values_col] = self.dataset[values_col]

            if values_col:
                values = self.dataset[values_col].sum(axis=1)  # Summing across selected columns for each row

                st.info("Successfully computed")
                # Calculate IRR
                resultant_dataframe["Internal Rate of Return"] = npf.irr(values)
                st.dataframe(resultant_dataframe)

        elif Type == "mirr":
            st.success("Provide dataset for MIRR calculation")

            values_col = st.multiselect("Select the column(s) for cash flow values", self.dataset.columns)
            finance_rate_col = st.selectbox("Select column for 'Finance Rate'", self.dataset.columns)
            reinvest_rate_col = st.selectbox("Select column for 'Reinvest Rate'", self.dataset.columns)

            resultant_dataframe[values_col] = self.dataset[values_col]
            resultant_dataframe[finance_rate_col] = self.dataset[finance_rate_col]
            resultant_dataframe[reinvest_rate_col] = self.dataset[reinvest_rate_col]

            if values_col and finance_rate_col and reinvest_rate_col:
                values = self.dataset[values_col].sum(axis=1)  # Summing across selected columns for each row
                finance_rate = np.array(self.dataset[finance_rate_col].values)
                reinvest_rate = np.array(self.dataset[reinvest_rate_col].values)

                st.info("Successfully computed")
                # Calculate MIRR
                resultant_dataframe["Modified IRR"] = npf.mirr(values, finance_rate, reinvest_rate)
                st.dataframe(resultant_dataframe)

        elif Type == "nper":
            st.success("Provide dataset for Number of Periodic Payments calculation")

            rate_col = st.selectbox("Select column for 'RATE'", self.dataset.columns)
            pmt_col = st.selectbox("Select column for 'PMT'", self.dataset.columns)
            pv_col = st.selectbox("Select column for 'PV'", self.dataset.columns)
            fv_col = st.selectbox("Select column for 'FV' (optional)", self.dataset.columns)

            resultant_dataframe[rate_col] = self.dataset[rate_col]
            resultant_dataframe[pv_col] = self.dataset[pv_col]
            resultant_dataframe[pmt_col] = self.dataset[pmt_col]
            resultant_dataframe[fv_col] = self.dataset[fv_col]

            if rate_col and pmt_col and pv_col:
                rate = np.array(self.dataset[rate_col].values)
                pmt = np.array(self.dataset[pmt_col].values)
                pv = np.array(self.dataset[pv_col].values)
                fv = np.array(self.dataset[fv_col].values) if fv_col else 0

                st.info("Successfully computed")
                # Calculate number of periods
                resultant_dataframe["Number of Periods"] = npf.nper(rate, pmt, pv, fv)
                st.dataframe(resultant_dataframe)

    def work_with_numbers(self,Type):
        if Type == "fv":
            st.write("Future Value Calculation:")
            rate = st.number_input("Enter the 'rate' value (e.g., 0.05):")
            nper = st.number_input("Enter the 'nper' value (number of periods):")
            pmt = st.number_input("Enter the 'pmt' value (payment per period):")
            pv = st.number_input("Enter the 'pv' value (present value):")
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return rate, nper, pmt, pv, when

        elif Type == "pv":
            st.write("Present Value Calculation:")
            rate = st.number_input("Enter the 'rate' value (e.g., 0.05):")
            nper = st.number_input("Enter the 'nper' value (number of periods):")
            pmt = st.number_input("Enter the 'pmt' value (payment per period):")
            fv = st.number_input("Enter the 'fv' value (future value):", value=0.0)
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return rate, nper, pmt, fv, when

        elif Type == "npv":
            st.write("Net Present Value Calculation:")
            rate = st.number_input("Enter the 'rate' value (e.g., 0.05):")
            values = st.text_input("Enter the cash flow values as a list (e.g., [100, 200, 300]):")
            values = np.array(eval(values))
            return rate, values

        elif Type == "pmt":
            st.write("Payment Calculation (Principal + Interest):")
            rate = st.number_input("Enter the 'rate' value (e.g., 0.05):")
            nper = st.number_input("Enter the 'nper' value (number of periods):")
            pv = st.number_input("Enter the 'pv' value (present value):")
            fv = st.number_input("Enter the 'fv' value (future value, optional):", value=0.0)
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return rate, nper, pv, fv, when

        elif Type == "ppmt":
            st.write("Principal Payment Calculation:")
            rate = st.number_input("Enter the 'rate' value (e.g., 0.05):")
            per = st.number_input("Enter the 'per' value (current period):")
            nper = st.number_input("Enter the 'nper' value (number of periods):")
            pv = st.number_input("Enter the 'pv' value (present value):")
            fv = st.number_input("Enter the 'fv' value (future value, optional):", value=0.0)
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return rate, per, nper, pv, fv, when

        elif Type == "ipmt":
            st.write("Interest Payment Calculation:")
            rate = st.number_input("Enter the 'rate' value (e.g., 0.05):")
            per = st.number_input("Enter the 'per' value (current period):")
            nper = st.number_input("Enter the 'nper' value (number of periods):")
            pv = st.number_input("Enter the 'pv' value (present value):")
            fv = st.number_input("Enter the 'fv' value (future value, optional):", value=0.0)
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return rate, per, nper, pv, fv, when

        elif Type == "irr":
            st.write("Internal Rate of Return Calculation:")
            values = st.text_input("Enter the cash flow values as a list (e.g., [-500, 200, 300, 400]):")
            values = np.array(eval(values))
            return values

        elif Type == "mirr":
            st.write("Modified Internal Rate of Return Calculation:")
            values = st.text_input("Enter the cash flow values as a list (e.g., [-500, 200, 300, 400]):")
            finance_rate = st.number_input("Enter the 'finance rate' (e.g., 0.05):")
            reinvest_rate = st.number_input("Enter the 'reinvest rate' (e.g., 0.07):")
            values = np.array(eval(values))
            return values, finance_rate, reinvest_rate

        elif Type == "nper":
            st.write("Number of Periods Calculation:")
            rate = st.number_input("Enter the 'rate' value (e.g., 0.05):")
            pmt = st.number_input("Enter the 'pmt' value (payment per period):")
            pv = st.number_input("Enter the 'pv' value (present value):")
            fv = st.number_input("Enter the 'fv' value (future value, optional):", value=0.0)
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return rate, pmt, pv, fv, when

        elif Type == "rate":
            st.write("Interest Rate Calculation:")
            nper = st.number_input("Enter the 'nper' value (number of periods):")
            pmt = st.number_input("Enter the 'pmt' value (payment per period):")
            pv = st.number_input("Enter the 'pv' value (present value):")
            fv = st.number_input("Enter the 'fv' value (future value):")
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            guess = st.number_input("Enter the 'guess' value (optional):", value=0.1)
            return nper, pmt, pv, fv, when, guess
    def work_with_arrays(self, Type):
        if Type == "fv":
            st.write("Future Value Calculation:")
            rate = st.text_input("Enter the 'rate' value (e.g., [0.05]):")
            nper = st.text_input("Enter the 'nper' value (number of periods, e.g., [10]):")
            pmt = st.text_input("Enter the 'pmt' value (payment per period, e.g., [100]):")
            pv = st.text_input("Enter the 'pv' value (present value, e.g., [1000]):")
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return np.array(eval(rate)), np.array(eval(nper)), np.array(eval(pmt)), np.array(eval(pv)), when

        elif Type == "pv":
            st.write("Present Value Calculation:")
            rate = st.text_input("Enter the 'rate' value (e.g., [0.05]):")
            nper = st.text_input("Enter the 'nper' value (number of periods, e.g., [10]):")
            pmt = st.text_input("Enter the 'pmt' value (payment per period, e.g., [100]):")
            fv = st.text_input("Enter the 'fv' value (future value, e.g., [1000]):")
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return np.array(eval(rate)), np.array(eval(nper)), np.array(eval(pmt)), np.array(eval(fv)), when

        elif Type == "npv":
            st.write("Net Present Value Calculation:")
            rate = st.text_input("Enter the 'rate' value (e.g., [0.05]):")
            values = st.text_input("Enter the cash flow values as a list (e.g., [[100, 200], [300, 400]]):")
            return np.array(eval(rate)), np.array(eval(values))

        elif Type == "pmt":
            st.write("Payment Calculation (Principal + Interest):")
            rate = st.text_input("Enter the 'rate' value (e.g., [0.05]):")
            nper = st.text_input("Enter the 'nper' value (number of periods, e.g., [10]):")
            pv = st.text_input("Enter the 'pv' value (present value, e.g., [1000]):")
            fv = st.text_input("Enter the 'fv' value (future value, optional, e.g., [0]):")
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return np.array(eval(rate)), np.array(eval(nper)), np.array(eval(pv)), np.array(ast.literal_eval(fv)), when

        elif Type == "ppmt":
            st.write("Principal Payment Calculation:")
            rate = st.text_input("Enter the 'rate' value (e.g., [0.05]):")
            per = st.text_input("Enter the 'per' value (current period, e.g., [1]):")
            nper = st.text_input("Enter the 'nper' value (number of periods, e.g., [10]):")
            pv = st.text_input("Enter the 'pv' value (present value, e.g., [1000]):")
            fv = st.text_input("Enter the 'fv' value (future value, optional, e.g., [0]):")
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return np.array(eval(rate)), np.array(eval(per)), np.array(eval(nper)), np.array(eval(pv)), np.array(ast.literal_eval(fv)), when

        elif Type == "ipmt":
            st.write("Interest Payment Calculation:")
            rate = st.text_input("Enter the 'rate' value (e.g., [0.05]):")
            per = st.text_input("Enter the 'per' value (current period, e.g., [1]):")
            nper = st.text_input("Enter the 'nper' value (number of periods, e.g., [10]):")
            pv = st.text_input("Enter the 'pv' value (present value, e.g., [1000]):")
            fv = st.text_input("Enter the 'fv' value (future value, optional, e.g., [0]):")
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return np.array(eval(rate)), np.array(eval(per)), np.array(eval(nper)), np.array(eval(pv)), np.array(eval(fv)), when

        elif Type == "irr":
            st.write("Internal Rate of Return Calculation:")
            values = st.text_input("Enter the cash flow values as a list (e.g., [[-500, 200], [300, 400]]):")
            return np.array(eval(values))

        elif Type == "mirr":
            st.write("Modified Internal Rate of Return Calculation:")
            values = st.text_input("Enter the cash flow values as a list (e.g., [[-500, 200], [300, 400]]):")
            finance_rate = st.text_input("Enter the 'finance rate' (e.g., [0.05]):")
            reinvest_rate = st.text_input("Enter the 'reinvest rate' (e.g., [0.07]):")
            return np.array(eval(values)), np.array(eval(finance_rate)), np.array(eval(reinvest_rate))

        elif Type == "nper":
            st.write("Number of Periods Calculation:")
            rate = st.text_input("Enter the 'rate' value (e.g., [0.05]):")
            pmt = st.text_input("Enter the 'pmt' value (payment per period, e.g., [100]):")
            pv = st.text_input("Enter the 'pv' value (present value, e.g., [1000]):")
            fv = st.text_input("Enter the 'fv' value (future value, optional, e.g., [0]):")
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return np.array(eval(rate)), np.array(eval(pmt)), np.array(eval(pv)), np.array(eval(fv)), when

        elif Type == "rate":
            st.write("Interest Rate Calculation:")
            nper = st.text_input("Enter the 'nper' value (number of periods, e.g., [10]):")
            pmt = st.text_input("Enter the 'pmt' value (payment per period, e.g., [100]):")
            pv = st.text_input("Enter the 'pv' value (present value, e.g., [1000]):")
            fv = st.text_input("Enter the 'fv' value (future value, e.g., [0]):")
            when = st.selectbox("When payments are due", [0, 1], help="0 for end of period, 1 for beginning")
            return np.array(eval(nper)), np.array(eval(pmt)), np.array(eval(pv)), np.array(eval(fv)),when           
            
    def main(self,Type):
        col1,col2=st.columns([1,1])
        with col1:
            index_column=st.checkbox("Apply with all indexes and columns")
            numbers=st.checkbox("Work With Numbers")
            arrays=st.checkbox("Work With Arrays")
        with col2:
            if index_column:
                st.info("Switched to all indexes and all columns")
                dataset=self.dataset
                self.work_with_datasets(Type,dataset)

            if numbers:
                st.info("Switched To Numbers")
                    
                if Type == "fv":
                    rate, nper, pmt, pv, when = self.work_with_numbers(Type)
                    result = npf.fv(rate, nper, pmt, pv, when)
                    st.write(f"Future Value: {result}")
                    st.write("Explanation: The future value is calculated based on the present value, payment amount, interest rate, and the number of periods.")

                elif Type == "pv":
                    rate, nper, pmt, fv, when = self.work_with_numbers(Type)
                    result = npf.pv(rate, nper, pmt, fv, when)
                    st.write(f"Present Value: {result}")
                    st.write("Explanation: The present value is determined by discounting the future cash flows back to their present value using the specified rate.")

                elif Type == "npv":
                    rate, values = self.work_with_numbers(Type)
                    result = npf.npv(rate, values)
                    st.write(f"Net Present Value: {result}")
                    st.write("Explanation: The net present value is the sum of the present values of cash flows, discounted at the given rate.")

                elif Type == "pmt":
                    rate, nper, pv, fv, when = self.work_with_numbers(Type)
                    result = npf.pmt(rate, nper, pv, fv, when)
                    st.write(f"Payment: {result}")
                    st.write("Explanation: The payment calculation determines the amount that needs to be paid periodically to cover the principal and interest.")

                elif Type == "ppmt":
                    rate, per, nper, pv, fv, when = self.work_with_numbers(Type)
                    result = npf.ppmt(rate, per, nper, pv, fv, when)
                    st.write(f"Principal Payment: {result}")
                    st.write("Explanation: The principal payment is the portion of the payment that goes towards paying down the loan principal.")

                elif Type == "ipmt":
                    rate, per, nper, pv, fv, when = self.work_with_numbers(Type)
                    result = npf.ipmt(rate, per, nper, pv, fv, when)
                    st.write(f"Interest Payment: {result}")
                    st.write("Explanation: The interest payment is the portion of the payment that goes towards paying the interest on the loan.")

                elif Type == "irr":
                    values = self.work_with_numbers(Type)
                    result = npf.irr(values)
                    st.write(f"Internal Rate of Return: {result}")
                    st.write("Explanation: The internal rate of return is the discount rate that makes the net present value of cash flows equal to zero.")

                elif Type == "mirr":
                    values, finance_rate, reinvest_rate = self.work_with_numbers(Type)
                    result = npf.mirr(values, finance_rate, reinvest_rate)
                    st.write(f"Modified Internal Rate of Return: {result}")
                    st.write("Explanation: The modified internal rate of return accounts for the cost of financing and the reinvestment rate of cash flows.")

                elif Type == "nper":
                    rate, pmt, pv, fv, when = self.work_with_numbers(Type)
                    result = npf.nper(rate, pmt, pv, fv, when)
                    st.write(f"Number of Periods: {result}")
                    st.write("Explanation: The number of periods calculation determines how many periods are needed to reach the desired future value.")

                elif Type == "rate":
                    nper, pmt, pv, fv, when, guess = self.work_with_numbers(Type)
                    result = npf.rate(nper, pmt, pv, fv, when, guess)
                    st.write(f"Interest Rate: {result}")
                    st.write("Explanation: The interest rate calculation finds the rate of interest per period necessary to achieve the future value with the given payments.")
                                    
            if arrays:
                st.info("Switched to arrays")
                if Type == "fv":
                    rate, nper, pmt, pv, when = self.work_with_arrays(Type)
                    result = npf.fv(rate, nper, pmt, pv, when)
                    st.write(f"Future Value: {result}")
                    st.write("Explanation: The future value is calculated based on the present value, payment amount, interest rate, and the number of periods.")

                elif Type == "pv":
                    rate, nper, pmt, fv, when = self.work_with_arrays(Type)
                    result = npf.pv(rate, nper, pmt, fv, when)
                    st.write(f"Present Value: {result}")
                    st.write("Explanation: The present value is determined by discounting the future cash flows back to their present value using the specified rate.")

                elif Type == "npv":
                    rate, values = self.work_with_arrays(Type)
                    result = np.npv(rate, values)
                    st.write(f"Net Present Value: {result}")
                    st.write("Explanation: The net present value is the sum of the present values of cash flows, discounted at the given rate.")

                elif Type == "pmt":
                    rate, nper, pv, fv, when = self.work_with_arrays(Type)
                    result = npf.pmt(rate, nper, pv, fv, when)
                    st.write(f"Payment: {result}")
                    st.write("Explanation: The payment calculation determines the amount that needs to be paid periodically to cover the principal and interest.")

                elif Type == "ppmt":
                    rate, per, nper, pv, fv, when = self.work_with_arrays(Type)
                    result = npf.ppmt(rate, per, nper, pv, fv, when)
                    st.write(f"Principal Payment: {result}")
                    st.write("Explanation: The principal payment is the portion of the payment that goes towards paying down the loan principal.")

                elif Type == "ipmt":
                    rate, per, nper, pv, fv, when = self.work_with_arrays(Type)
                    result = npf.ipmt(rate, per, nper, pv, fv, when)
                    st.write(f"Interest Payment: {result}")
                    st.write("Explanation: The interest payment is the portion of the payment that goes towards paying the interest on the loan.")

                elif Type == "irr":
                    values = self.work_with_arrays(Type)
                    result = npf.irr(values)
                    st.write(f"Internal Rate of Return: {result}")
                    st.write("Explanation: The internal rate of return is the discount rate that makes the net present value of cash flows equal to zero.")

                elif Type == "mirr":
                    values, finance_rate, reinvest_rate = self.work_with_arrays(Type)
                    result = npf.mirr(values, finance_rate, reinvest_rate)
                    st.write(f"Modified Internal Rate of Return: {result}")
                    st.write("Explanation: The modified internal rate of return accounts for the cost of financing and the reinvestment rate of cash flows.")

                elif Type == "nper":
                    rate, pmt, pv, fv, when = self.work_with_arrays(Type)
                    result = npf.nper(rate, pmt, pv, fv, when)
                    st.write(f"Number of Periods: {result}")
                    st.write("Explanation: The number of periods calculation determines how many periods are needed to reach the desired future value.")

                elif Type == "rate":
                    nper, pmt, pv, fv, when = self.work_with_arrays(Type)
                    result = npf.rate(nper, pmt, pv, fv, when)
                    st.write(f"Interest Rate: {result}")
                
class LinearAlgebra:
    def __init__(self,dataframe):
        self.dataset=dataset
    def layout(self):
        st.sidebar.info("Switched to linear algebra")
        dot=st.sidebar.checkbox("Apply DOT Product")
        if dot:
            self.main("dot")
        vdot=st.sidebar.checkbox("Apply VDOT Product")
        if vdot:
            self.main("Vdot")
        inner=st.sidebar.checkbox("Apply INNER Product")
        if inner:
            self.main("inner")
        outter=st.sidebar.checkbox("Apply OUTTER Product")
        if outter:
            self.main("outer")
        kron=st.sidebar.checkbox("Apply Kronicker Product")
        if kron:
            self.main("kron")
        cholesky=st.sidebar.checkbox("Apply Cholesky Decomposition")
        if cholesky:
            self.main("cholesky")
        qr=st.sidebar.checkbox("Apply Qr Decomposition")
        if qr:
            self.main("qr")
        svd=st.sidebar.checkbox("Apply SVD Decomposition")
        if svd:
            self.main("svd")
        eigen_values=st.sidebar.checkbox("Fing Eigen values and Eigen Vectors")
        if eigen_values:
            self.main("eigen values and vectors")
        eigh=st.sidebar.checkbox("Find Eigen Values And Eigen Vectors For Hermittian Matrix")
        if eigh:
            self.main("hermetian eigen values and vectors")
        norm=st.sidebar.checkbox("Find Norm")
        if norm:
            self.main("norm")
        cond=st.sidebar.checkbox("Compute the condition number of a matrix")
        if cond:
            self.main("cond")
        det=st.sidebar.checkbox(" Compute the determinant of an array.")
        if dot:
            self.main("det")
        slogdet=st.sidebar.checkbox(" Compute the sign and (natural) logarithm of the determinant of an array.")
        if slogdet:
            self.main("slogdet")
        trace=st.sidebar.checkbox("Trace of the matrix")
        if trace:
            self.main("trace")
        matrix_power=st.sidebar.checkbox("Find The Matrix Power")
        if matrix_power:
            self.main("matrix_power")
    def main(self,Type):
# nothing bhere
        df=pd.DataFrame()
        if Type in ["dot","Vot","inner","outter","kron"]:
            rows_selected=st.multiselect("Please select Your Desired Rows",self.dataset.index)
            if rows_selected:
                new_dataset=self.dataset.loc[rows_selected,:]
            else:
                new_dataset=self.dataset
            col1,col2=st.columns([1,1])
            with col1:
                column1=st.selectbox("Select first list",new_dataset.columns)
            with col2:
                column2=st.selectbox("Select second list",new_dataset.columns)
            if column1 and column2:
                st.info("Your Data Frame")
                st.dataframe(new_dataset[[column1,column2]])
                if Type=="dot":
                    column1=np.array(list(new_dataset[column1].values))
                    column2=np.array(list(new_dataset[column2].values))
                    dot_value=np.dot(column1,column2)
                    st.subheader(f"YOUR RESULT (Dot): {dot_value}")
                if Type=="Vdot":
                    column1=np.array(list(new_dataset[column1].values))
                    column2=np.array(list(new_dataset[column2].values))
                    vdot_value=np.vdot(column1,column2)
                    st.subheader(f"YOUR RESULT : {vdot_value}")
                if Type=="inner":
                    column1=np.array(list(new_dataset[column1].values))
                    column2=np.array(list(new_dataset[column2].values))
                    inner_product=np.inner(column1,column2)
                    st.subheader(f"YOUR RESULT (Inner): {inner_product}")
                if Type=="outer":
                    column1=np.array(list(new_dataset[column1].values))
                    column2=np.array(list(new_dataset[column2].values))
                    outer_product=np.outer(column1,column2)
                    st.subheader(f"YOUR RESULT (Outter)")
                    st.table(outer_product)
                if Type=="kron":
                    column1=np.array(list(new_dataset[column1].values))
                    column2=np.array(list(new_dataset[column2].values))
                    kron_product=np.kron(column1,column2)
                    st.info("Your Result : (Kronicker)")
                    st.text(np.array(kron_product))
        if Type in ["cholesky", "qr", "svd"]:
            selected_rows = st.multiselect("Please enter the desired rows", self.dataset.index)            
            new_dataset = self.dataset.loc[selected_rows, :] if selected_rows else self.dataset
            col1,col2=st.columns([1,1])
            with col1:
                list1 = col1.multiselect("Please select columns (for list1)", self.dataset.columns)
            with col2:
                list2 = col2.multiselect("Please select columns (for list2)", self.dataset.columns)
            if list1 and list2:
                combined_columns = list(set(list1 + list2))
                with col1:
                    st.info("Your Data Frame")
                    st.dataframe(new_dataset[list1])
                with col2:
                    st.info("Your Data Frame")
                    st.dataframe(new_dataset[list2])
                array1 = new_dataset[list1].to_numpy()
                array2 = new_dataset[list2].to_numpy()
                if Type == "cholesky":
                    try:
                        if array1.shape[1] == array1.shape[0]:
                            cholesky_decomposition = np.linalg.cholesky(array1)
                            st.write("Cholesky Decomposition Result:")
                            st.write(cholesky_decomposition)
                        else:
                            st.error("Cholesky decomposition requires a square matrix.")
                    except np.linalg.LinAlgError as e:
                        st.error(f"Cholesky decomposition error: {e}")

                elif Type == "qr":
                    q, r = np.linalg.qr(array1)
                    st.write("QR Decomposition Result:")
                    st.write("Q matrix:", q)
                    st.write("R matrix:", r)

                elif Type == "svd":
                    u, s, vh = np.linalg.svd(array1, full_matrices=False)
                    st.write("SVD Decomposition Result:")
                    st.write("U matrix:", u)
                    st.write("Singular values:", s)
                    st.write("V^T matrix:", vh)
        if Type in ["eigen values and vectors", "hermetian eigen values and vectors"]:
            # Select rows
            selected_rows = st.multiselect("Please enter the desired rows", self.dataset.index)
            
            # If rows are selected, filter the dataset; otherwise, use the full dataset
            new_dataset = self.dataset.loc[selected_rows, :] if selected_rows else self.dataset
            list1 = st.multiselect("Please select columns (for list1)", self.dataset.columns)
            if list1:
                st.info("Your Data Frame")
                st.dataframe(new_dataset[list1])
                array1 = new_dataset[list1].to_numpy()
                if Type == "eigen values and vectors":
                    try:
                        eigenvalues, eigenvectors = np.linalg.eig(array1)
                        st.write("Eigenvalues:", eigenvalues)
                        st.write("Eigenvectors:", eigenvectors)
                    except np.linalg.LinAlgError as e:
                        st.error(f"Error calculating eigenvalues and eigenvectors: {e}")

                elif Type == "hermetian eigen values and vectors":
                    try:
                        if np.allclose(array1, array1.T.conj()):
                            eigenvalues, eigenvectors = np.linalg.eigh(array1)
                            st.write("Hermitian Eigenvalues:", eigenvalues)
                            st.write("Hermitian Eigenvectors:", eigenvectors)
                        else:
                            st.error("The matrix is not Hermitian. A Hermitian matrix is required for this operation.")
                    except np.linalg.LinAlgError as e:
                        st.error(f"Error calculating Hermitian eigenvalues and eigenvectors: {e}")
        if Type in ["norm", "det", "matrix_power", "cond", "slogdet", "trace"]:
            selected_rows = st.multiselect("Please enter the desired rows", self.dataset.index)
            new_dataset = self.dataset.loc[selected_rows, :] if selected_rows else self.dataset
            list1 = st.multiselect("Please select columns (for list1)", self.dataset.columns)
            if list1:
                st.info("Your Data Frame")
                st.dataframe(new_dataset[list1])
                array1 = new_dataset[list1].to_numpy()
                if Type == "norm":
                    ord_value = st.selectbox("Select norm order", [None, 'fro', 'nuc', np.inf, -np.inf, 0, 1, -1, 2, -2], index=0)
                    axis_value = st.selectbox("Select axis", [None, 0, 1, (0, 1)], index=0)
                    keepdims_value = st.checkbox("Keep dimensions (keepdims)", value=False)
                    norm_result = np.linalg.norm(array1, ord=ord_value, axis=axis_value, keepdims=keepdims_value)
                    st.write(f"Norm Result (ord={ord_value}, axis={axis_value}, keepdims={keepdims_value}):")
                    st.write(norm_result)

                elif Type == "det":
                    try:
                        det_result = np.linalg.det(array1)
                        st.write("Determinant Result:")
                        st.write(det_result)
                    except np.linalg.LinAlgError as e:
                        st.error(f"Error calculating determinant: {e}")
                

                elif Type == "matrix_power":
                    power_value = st.number_input("Enter the power (integer value)", value=2, step=1)
                    try:
                        matrix_power_result = np.linalg.matrix_power(array1, power_value)
                        st.write(f"Matrix powered to {power_value}:")
                        st.write(matrix_power_result)
                    except np.linalg.LinAlgError as e:
                        st.error(f"Error calculating matrix power: {e}")

                elif Type == "cond":
                    p_value = st.selectbox("Select p-norm type", [None, 1, -1, 2, -2, np.inf, -np.inf, 'fro', 'nuc'], index=0)
                    try:
                        cond_result = np.linalg.cond(array1, p=p_value)
                        st.write(f"Condition Number (p={p_value}):")
                        st.write(cond_result)
                    except np.linalg.LinAlgError as e:
                        st.error(f"Error calculating condition number: {e}")

                elif Type == "slogdet":
                    try:
                        sign, logdet = np.linalg.slogdet(array1)
                        st.write("Sign of Determinant:")
                        st.write(sign)
                        st.write("Log of Absolute Determinant:")
                        st.write(logdet)
                    except np.linalg.LinAlgError as e:
                        st.error(f"Error calculating sign and log determinant: {e}")

                elif Type == "trace":
                    offset_value = st.number_input("Enter offset (default=0)", value=0, step=1)
                    axis1_value = st.selectbox("Select axis1 (default=0)", [0, 1], index=0)
                    axis2_value = st.selectbox("Select axis2 (default=1)", [0, 1], index=1)
                    try:
                        trace_result = np.trace(array1, offset=offset_value, axis1=axis1_value, axis2=axis2_value)
                        st.write(f"Trace Result (offset={offset_value}, axis1={axis1_value}, axis2={axis2_value}):")
                        st.write(trace_result)
                    except Exception as e:
                        st.error(f"Error calculating trace: {e}")


        
#Defining the main layout
file=st.sidebar.file_uploader("Upload Any CSV File",["csv"])
if file:
    # Reading the uploaded file as bytes (file-like object)
    csv_bytes = file.read(100000)  # Read the first 100KB to detect encoding
    result = chardet.detect(csv_bytes)
    encoding = result['encoding']

    # Move back to the start of the file after reading
    file.seek(0)

    # Read CSV using the detected encoding
    dataset = pd.read_csv(file, encoding=encoding)
    with st.sidebar:
        options=option_menu("Choose The Functionality",["Basic Mathematical Operations","Descriptive Statistics","Linear Algebra","Trignometric Functions","Financial Functions"],orientation="horizontal")

    if options=="Descriptive Statistics":
        Descriptive(dataset).layout()
    elif options=="Basic Mathematical Operations":
        BasicMathematicalOperations(dataset)._layout_()
    elif options=="Trignometric Functions":
        Trigonometry(dataset).layout()
    elif options=="Financial Functions":
        Financial_Functions(dataset).layout()
    elif options=="Linear Algebra":
        LinearAlgebra(dataset).layout()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="YBRFSS Question Sankey Visualization", layout="wide")

if 'df_full' not in st.session_state:
    st.warning("Please upload the data on the Home page first.")
    st.stop()

df = st.session_state['df_full']

def preprocess_suicide_data(df):
    df_suicide = df[['q26', 'q27', 'q28', 'q29', 'q30']].copy()
    df_suicide_imputed = df_suicide.dropna(how='all')

    # Rule 1: For Q28 that is yes, impute all Q27 null values to yes
    df_suicide_imputed.loc[df_suicide_imputed['q28'] == 1, 'q27'] = df_suicide_imputed.loc[df_suicide_imputed['q28'] == 1, 'q27'].fillna(1)

    # Rule 2: For Q29, any answer that attempted more than 0 times will impute the previous Q28 and Q29 to yes
    # Only impute q28 where it is NaN
    condition_q29 = df_suicide_imputed['q29'] > 0
    # q28_nan_condition = df_suicide_imputed['q28'].isna()
    q27_nan_condition = df_suicide_imputed['q27'].isna()
    # df_suicide_imputed.loc[condition_q29 & q28_nan_condition, 'q28'] = 1
    df_suicide_imputed.loc[condition_q29 & q27_nan_condition, 'q27'] = 1

    # Rule 3: For Q30, any answer that has a yes or no in injury, will impute Q27 and Q28 to yes
    # Only impute q27 and q28 where they are NaN
    condition_q30 = df_suicide_imputed['q30'] > 1
    q27_nan_condition = df_suicide_imputed['q27'].isna()
    # q28_nan_condition = df_suicide_imputed['q28'].isna()
    df_suicide_imputed.loc[condition_q30 & q27_nan_condition, 'q27'] = 1
    # df_suicide_imputed.loc[condition_q30 & q28_nan_condition, 'q28'] = 1

    # Rule 4: For Q30, any answer that has a yes in injury, will impute Q29 to 1 time
    # Only impute q29 where they are NaN
    condition_q30 = df_suicide_imputed['q30'] == 2
    q29_nan_condition = df_suicide_imputed['q29'].isna()
    df_suicide_imputed.loc[condition_q30 & q29_nan_condition, 'q29'] = 2

    # Rule 5: If Q27 is no, then Q28-Q30 all the null become no
    condition_q27 = df_suicide_imputed['q27'] == 2
    q28_nan_condition = df_suicide_imputed['q28'].isna()
    q29_nan_condition = df_suicide_imputed['q29'].isna()
    q30_nan_condition = df_suicide_imputed['q30'].isna()
    df_suicide_imputed.loc[condition_q27 & q28_nan_condition, 'q28'] = 2
    df_suicide_imputed.loc[condition_q27 & q29_nan_condition, 'q29'] = 1
    df_suicide_imputed.loc[condition_q27 & q30_nan_condition, 'q30'] = 1
    
    # Rule 6: If Q29 is no, then Q30 all the null is no suicide attempt
    condition_q29 = df_suicide_imputed['q29'] == 1
    q30_nan_condition = df_suicide_imputed['q30'].isna()
    df_suicide_imputed.loc[condition_q29 & q30_nan_condition, 'q30'] = 1

    # Sanity Check 1: Q28 is Yes but Q27 is No
    condition1 = (df_suicide_imputed['q28'] == 1) & (df_suicide_imputed['q27'] == 2)

    # Sanity Check 2: Q29 > 1 but Q27 is No
    condition2 = (df_suicide_imputed['q29'] > 1) & (df_suicide_imputed['q27'] == 2)

    # Sanity Check 3: Q30 is Yes but Q27 is No
    condition3 = (df_suicide_imputed['q30'] == 2) & (df_suicide_imputed['q27'] == 2)

    # Sanity Check 4: Q29 is 0 times but Q30 is Yes
    condition4 = (df_suicide_imputed['q29'] == 1) & (df_suicide_imputed['q30'] == 2)

    # Combine all conditions
    invalid_rows = (
        condition1 | condition2 | condition3 | condition4
    )

    df_suicide_imputed_checked = df_suicide_imputed[~invalid_rows].copy()

    return df_suicide_imputed_checked

def rename_to_text(df_suicide):
    df_suicide_text =df_suicide.copy()
    mapping = {
        "q26": {1.0: "Yes", 2.0: "No", np.nan: "Null"},
        "q27": {1.0: "Yes", 2.0: "No", np.nan: "Null"},
        "q28": {1.0: "Yes", 2.0: "No", np.nan: "Null"},
        "q29":{
        1.0: "0 times",
        2.0: "1 time",
        3.0: "2 or 3 times",
        4.0: "4 or 5 times",
        5.0: "6 or more times",
        np.nan: "Null",  # Handle NaN values
    },
        "q30": {1.0: "No suicide attempts", 2.0: "Yes", 3.0: "No", np.nan: "Null"},
    }
    df_suicide_text = df_suicide_text.replace(mapping)
    df_suicide_text = df_suicide_text.rename(columns={'q26':'Q26 Depressed','q27':'Q27 Considered', 'q28':'Q28 Plan', 'q29':'Q29 Attempted', 'q30':'Q30 Injurious'})
    return df_suicide_text

def create_sankey_data(df, name):
    sankey_data = rename_to_text(df)
    detailed_labels = [f"{col}: {value}" for col in sankey_data.columns for value in sankey_data[col].unique()]
    label_indices = {label: idx for idx, label in enumerate(detailed_labels)}

    sources_detailed, targets_detailed, values_detailed = [], [], []

    for i in range(len(sankey_data.columns) - 1):
        col1, col2 = sankey_data.columns[i], sankey_data.columns[i + 1]
        flow_counts = sankey_data.groupby([col1, col2]).size().reset_index(name='count')
        for _, row in flow_counts.iterrows():
            sources_detailed.append(label_indices[f"{col1}: {row[col1]}"])
            targets_detailed.append(label_indices[f"{col2}: {row[col2]}"])
            values_detailed.append(row['count'])

    fig_detailed = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[f"{col}: {value}" for col in sankey_data.columns for value in sankey_data[col].unique()],
        ),
        link=dict(
            source=sources_detailed,
            target=targets_detailed,
            value=values_detailed
        ),
        textfont=dict(color='black', size=15)
    )])

    fig_detailed.update_layout(title_text=f"Sankey Diagram: {name}", font_size=15, height=600)

    return fig_detailed

st.title("Adolescent Suicide Data Preprocessing and Sankey Visualization")

def preprocess_full_data(df):
    df_suicide_clean_whole = st.session_state['df_full'].iloc[df.index]

    # List of relevant columns
    relevant_columns = [
        # Stats info
        'weight', 'sitename',

        # Demographics
        'age', 'sex', 'grade', 'race4', 'race7',

        # Mental Health
        'q84', 'q26', 'q27', 'q28', 'q29', 'q30',

        # Adverse Childhood Experiences
        'qemoabuseace', 'qphyabuseace', 'qsexabuseace', 'qverbalabuseace', 'qphyviolenceace', 'qtreatbadlyace',
        'qlivedwillace','qlivedwabuseace', 'qintviolenceace', 'qunfairlyace',


        # Substance Use
        'q42', 'q46', 'q49', 'q50', 'q51', 'q52', 'qcurrentopioid', 'qhallucdrug',

        # Social and School Safety
        'q24', 'q25', 'q14', 'q15', 'qclose2people', 'qtalkadultace', 'qtalkfriendace'
    ]

    df_suicide_clean_whole = df_suicide_clean_whole[relevant_columns]
    df2 =df_suicide_clean_whole.copy()
    df2['Label'] = 1
    df2.loc[df2['q27'] == 1, 'Label'] = 2
    df2.loc[df2['q29'] >= 2, 'Label'] = 3
    df2.drop(columns=['q26','q27', 'q29', 'q28', 'q30'], inplace=True)

    # Calculate the ratio of missing values for each column
    nan_ratio = df2.isnull().mean()
    cutoff = 0.5
    drop_cols = nan_ratio[nan_ratio > cutoff].index
    df3 = df2.drop(columns=drop_cols)
    nan_ratio = df3.isnull().mean()
    return df3.drop(['q50','q15', 'race4'], axis=1).dropna().reset_index(drop=True)


if 'df_full' in st.session_state:
    df_clean = preprocess_suicide_data(st.session_state['df_full'])
    df_clean.dropna(inplace=True, subset=['q26','q27', 'q29','q30'])
    sankey_fig = create_sankey_data(df_clean, "Clean Suicide Data")

    st.plotly_chart(sankey_fig, use_container_width=True)
    df_clean_whole = preprocess_full_data(df_clean)
    st.dataframe(df_clean_whole)
    st.session_state['df_full_whole'] = df_clean_whole



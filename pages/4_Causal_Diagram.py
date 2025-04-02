import pandas as pd
import numpy as np
from dowhy import CausalModel
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from imblearn.over_sampling import SMOTE
import networkx as nx
import os

html_path = "causal_graph.html"

st.set_page_config(page_title="YBRFSS Question Causal Analysis", layout="wide")
st.title("Cansual Analysis and Visualizations")

if not os.path.exists(html_path):
    if 'df_full_whole' not in st.session_state:
        st.warning("Please preprocess the data in the second page first.")
        st.stop()

    data = st.session_state['df_full_whole']

    # Rename columns for readability
    mapping2 = {
        "q25": "Q25 - Electronically bullied",
        "q24": "Q24 - School bullied",
        "q84": "Q84 - Mental health",
        "q49": "Q49 - Prescription pill",
        "q52": "Q52 - Heroin taken",
        "q42": "Q42 - Alcohol",
        "q14": "Q14 - School unsafe",
        "qclose2people": "Q - Close to people",
        "sex": "Sex",
        "race7": "Race",
        "grade": "Grade",
        "age": "Age",
        "sitename": "Location",
    }

    data_cleaned = data.rename(columns=mapping2)

    feature_groups = {
        "Demographics": ["Age", "Sex", "Grade", "Race"],
        "Mental and Behavioral": [
            "Q84 - Mental health",
            "Q42 - Alcohol",
            "Q49 - Prescription pill",
            "Q52 - Heroin taken",
        ],
        "Bullying and Safety": [
            "Q24 - School bullied",
            "Q25 - Electronically bullied",
            "Q14 - School unsafe",
            "Q - Close to people",
        ],
    }

    outcome = "Label"
    causal_analysis_data = data_cleaned[
        feature_groups["Demographics"]
        + feature_groups["Mental and Behavioral"]
        + feature_groups["Bullying and Safety"]
        + [outcome, "weight"]
    ]

    # Apply row replication based on weights
    causal_analysis_data_weighted = causal_analysis_data.loc[
        causal_analysis_data.index.repeat(causal_analysis_data["weight"].astype(int))
    ]

    # Apply SMOTE
    X = causal_analysis_data_weighted.drop(columns=["Label", "weight"])
    y = causal_analysis_data_weighted["Label"]
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    causal_analysis_data_weighted = pd.concat([X_balanced, y_balanced], axis=1)

    # Run causal estimation for all treatments
    results = []
    treatments = feature_groups["Bullying and Safety"] + feature_groups["Mental and Behavioral"]
    common_causes = feature_groups["Demographics"] + [
        "Q84 - Mental health",
        "Q42 - Alcohol",
        "Q49 - Prescription pill"
    ]

    for treatment in treatments:
        causes = [c for c in common_causes if c != treatment]
        model = CausalModel(
            data=causal_analysis_data_weighted,
            treatment=treatment,
            outcome="Label",
            common_causes=causes
        )
        identified_estimand = model.identify_effect()
        causal_effect = model.estimate_effect(
            identified_estimand, method_name="backdoor.linear_regression"
        )
        results.append((treatment, causal_effect.value))

    # Build graph
    G = nx.DiGraph()
    for treatment, effect in results:
        G.add_node(treatment, node_type="treatment", subset=2)
        G.add_edge(treatment, "Suicide", weight=effect, edge_type="treatment_effect")

    for cause in common_causes:
        G.add_node(cause, node_type="common_cause", subset=0)
        for treatment, _ in results:
            if cause != treatment:
                G.add_edge(cause, treatment, weight=0, edge_type="common_cause")

    G.add_node("Suicide", node_type="outcome", subset=1)

    # Visualize with PyVis
    net = Network(height='800px', width='100%', bgcolor='#ffffff', font_color='black', directed=True)

    for node, data in G.nodes(data=True):
        label = node
        color = "lightblue"
        shape = "dot"
        if data["node_type"] == "common_cause":
            color = "lightgreen"
        elif data["node_type"] == "outcome":
            color = "orange"
        elif data["node_type"] == "both":
            shape = "box"
        net.add_node(node, label=label, color=color, shape=shape)

    for u, v, d in G.edges(data=True):
        label = f"{abs(d['weight']):.2f}" if d['weight'] != 0 else ""
        color = "red" if d["edge_type"] == "treatment_effect" else "gray"
        dashes = False if d["edge_type"] == "treatment_effect" else True
        net.add_edge(u, v, value=abs(d['weight']), title=label, label=label, color=color, arrows='to', dashes=dashes)

    net.set_options("""
    {
    "nodes": {
        "shape": "dot",
        "size": 20,
        "font": {
        "size": 18
        }
    },
    "edges": {
        "smooth": false,
        "arrows": {
        "to": {
            "enabled": true,
            "scaleFactor": 1.1
        }
        }
    },
    "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
        "gravitationalConstant": -550,
        "centralGravity": 0.2,
        "springLength": 500,
        "springConstant": 0.02,
        "damping": 0.4
        },
        "minVelocity": 0.5,
        "stabilization": {
        "enabled": true,
        "iterations": 1500
        }
    }
    }
    """)

    net.save_graph(html_path)
HtmlFile = open(html_path, 'r', encoding='utf-8')
components.html(HtmlFile.read(), height=1000)

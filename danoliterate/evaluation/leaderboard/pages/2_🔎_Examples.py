import pandas as pd
import streamlit as st

from danoliterate.infrastructure.constants import REPO_PATH

st.set_page_config("Danoliterate Examples", page_icon="🇩🇰")
# https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/17
hide_streamlit_style = """
        <style>
        [data-testid="stToolbar"] {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Danoliterate GLLM Examples")
st.warning("The benchmark is a beta version and results are subject to change.", icon="🤖")
"""
Inspect some
"""
scenarios = {
    scenario.stem.replace(".csv", ""): pd.read_csv(scenario, index_col=0)
    for scenario in sorted(
        (REPO_PATH / "danoliterate" / "evaluation" / "leaderboard" / "pages" / "assets")
        .resolve()
        .glob("*.csv")
    )
}
chosen_scenario = st.selectbox("Scenario", scenarios)
data = scenarios[chosen_scenario]
chosen_model = st.selectbox("Model", [col for col in data.columns if col != "prompt"])
if st.button("Show output examples"):
    for label, row in data.iterrows():
        st.markdown(f"### Prompt {label}")
        st.code(row["prompt"], language=None)
        st.markdown(f"### {chosen_model} Generation {label}")
        st.code(row[chosen_model], language=None)
        st.divider()

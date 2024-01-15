"""
Should be run as streamlit application
"""

import streamlit as st

st.set_page_config("Danoliterate Benchmark", page_icon="🇩🇰")
hide_streamlit_style = """
        <style>
        [data-testid="stToolbar"] {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Danoliterate GLLMs")
st.warning("The benchmark is a beta version and results are subject to change.", icon="🤖")

"""
## What is this?
This site presents the :sparkles: Danoliterate Generative Large Language Model Benchmark :sparkles:, evaluating how well models like ChatGPT, LlaMa or Mistral perform in Danish.
## Where can I see it?
Press `leaderboard` in the left sidebar to see how the models were ranked.
To inspect some specific examples of what the models generate, press `examples`.
## How can I learn more?
Currently, the main documentation for this benchmark is the Master's Thesis
[''Are GLLMs Danoliterate? Benchmarking Generative NLP in Danish''](https://sorenmulli.github.io/thesis/thesis.pdf).
The implementation is open and can be found on [sorenmulli/danoliterate](https://github.com/sorenmulli/danoliterate):
Please follow along and participate!
## Who made this?
This is part of a Master's Thesis produced by Søren Vejlgaard Holm at the DTU Compute in collaboration with [Alvenir](https://www.alvenir.ai/).
It was supervised by
- Lars Kai Hansen, DTU Compute

- Martin Carsten Nielsen, Alvenir

The work was supported by the Pioneer Centre for AI.
"""

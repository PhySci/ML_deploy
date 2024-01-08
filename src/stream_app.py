import streamlit as st
from ml import inference
from collections import namedtuple


def get_prediction():
    params = [[st.session_state.popularity, st.session_state.danceability, st.session_state.energy, st.session_state.key,
               st.session_state.loudness, st.session_state.mode, st.session_state.speechness, st.session_state.acousticness,
               st.session_state.instrumentalness, st.session_state.liveness, st.session_state.valence,
               st.session_state.tempo, st.session_state.duration*60*1000, st.session_state.tsig]]
    y_pr = inference(params)[0]
    st.write("Predicted class is ", y_pr[0])


slider_params = namedtuple("slider_params",
                           ["key", "min_value", "max_value", "step", "label"],
                           defaults=[0.0, 1.0, 0.05, None])

sliders = [
    slider_params("popularity", 0, 100, 1),
    slider_params("danceability"),
    slider_params("energy"),
    slider_params("key", 1, 11, 1),
    slider_params("loudness", -40, 2, 1),
    slider_params("mode", 0, 10, 1),
    slider_params("speechness"),
    slider_params("acousticness"),
    slider_params("instrumentalness"),
    slider_params("liveness"),
    slider_params("valence"),
    slider_params("tempo", 30, 220, 1),
    slider_params("duration", 0.1, 25.0, 0.1),
    slider_params("tsig", 1, 5, 1)
]

with st.sidebar:
    st.write("Введите параметры трека")

col1, col2 = st.columns(2)


with col1:
    for slider in sliders[:7]:
        st.slider(slider.label or slider.key,
                  slider.min_value,
                  slider.max_value,
                  slider.step,
                  key=slider.key,
                  on_change=get_prediction)

with col2:
    for slider in sliders[7:]:
        st.slider(slider.label or slider.key,
                  slider.min_value,
                  slider.max_value,
                  slider.step,
                  key=slider.key,
                  on_change=get_prediction)
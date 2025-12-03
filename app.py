import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

dist_options = {
    "Alpha": stats.alpha,
    "Beta": stats.beta,
    "Chi": stats.chi,
    "Exponential": stats.expon,
    "Lognormal": stats.lognorm,
    "Normal": stats.norm,
    "Gamma": stats.gamma,
    "Pareto": stats.pareto,
    "Rayleigh": stats.rayleigh,
    "Uniform": stats.uniform
    }

st.title("Histogram Fitting Tool 📊")
st.write("This app allows users to automatically fit distributions according to uploaded data and manually adjust parameters. 10 different data distributions are available.")

with st.sidebar:
    st.title("Input data here")
    
    choice = st.radio("Choose input method", ["Manual Entry", "CSV Upload"])
    data = None
    if choice == "Manual Entry":
        user_input = st.text_area("Enter data values seperated by commas:", "1, 2, 3 , 4 , 5")
        try:
            data = np.array([float(x) for x in user_input.split(",")])
        except:
            st.error("Invalid input, please enter numeric values and seperate by commas.")
    else:
        file = st.file_uploader("Upload your CSV file with one numeric column")
        if file: 
            df = pd.read_csv(file)
            col = st.selectbox("Select column:", df.columns)
            data = df[col].dropna().to_numpy()
            
    if data is None or len(data) < 5:
        st.error("Please upload or enter sufficient data to proceed.")
        st.stop()
        
    st.header("Distribution Options")
    dist_choice = st.selectbox("Select a distribution:", list(dist_options.keys()))
    dist = dist_options[dist_choice]
    
    
with st.expander("Fit Results"):
    param = dist.fit(data)
    param = tuple(float(p) for p in param)
    st.write(f"**Fitted Parameters:** {param}")
    
    fitted_dist = dist(*param)

    xs = np.linspace(min(data), max(data), 200)
    pdf_vals = fitted_dist.pdf(xs)

    hist_vals, bin_edges = np.histogram(data, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mean_sq_err = np.mean((np.interp(bin_centers, xs, pdf_vals) - hist_vals) ** 2)
    max_err = np.max(np.abs(np.interp(bin_centers, xs, pdf_vals) - hist_vals))

    st.write(f"**Mean Squared Error:** {mean_sq_err:.5f}")
    st.write(f"**Max Error:** {max_err:.5f}")
    
with st.sidebar.expander("Manual Parameter Adjustment"):
    manual = st.checkbox("Enable manual fitting")
    if manual:
        slider = []
        for i, p in enumerate(param):
            slider.append(
                st.slider(f"Parameter {i} (initial: {p:.3f})",
                min_value=float(p * 0.2),
                max_value=float(p * 5),
                value=float(p),
                )
            )
        manual_params = tuple(slider)
        fitted_dist = dist(*manual_params)
        manual_params = tuple(float(p) for p in slider)
        display_params = manual_params
    else:
        display_params = param
            
st.header("Data & Fitted Distribution 📈")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=30, density=True, alpha=0.5, label="Data", color="powderBlue")

xs = np.linspace(min(data), max(data), 300)
ax.plot(xs, fitted_dist.pdf(xs), "r-", label="Fitted PDF", color="#473671")

ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()

st.pyplot(fig)

with st.sidebar:
    st.header("Summary")
    st.write(f"Selected Distribution: **{dist_choice}**")
    st.write(f"Data Count: **{len(data)}**")
    st.write("Fitted Parameters:")
    st.write(display_params)  


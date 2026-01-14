import streamlit as st
import pandas as pd
from customer_journey_system import CustomerJourneySystem

st.set_page_config(
    page_title="Customer Journey AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Customer Journey AI System")
st.markdown("### AI-Powered Opportunity Prediction & Recommendation Engine")
st.markdown("---")

@st.cache_resource
def load_system():
    sys = CustomerJourneySystem("data_all.xltx")
    sys.train_win_probability_model()
    return sys

with st.spinner('Initializing System & Training Model...'):
    try:
        system = load_system()
        st.success("System Ready & Connected")
    except Exception as e:
        st.error(f"System Error: {e}")
        st.stop()

st.sidebar.header("Opportunity Parameters")

total_activities = st.sidebar.number_input("Total Activities", min_value=1, value=5)
unique_activities = st.sidebar.number_input("Unique Activities", min_value=1, value=3)

all_actions = sorted(system.df['action'].unique().tolist())
last_action = st.sidebar.selectbox("Last Action", all_actions)

all_countries = sorted(system.df['country'].unique().tolist())
country = st.sidebar.selectbox("Country", all_countries)

all_solutions = sorted(system.df['solution'].unique().tolist())
solution = st.sidebar.selectbox("Solution", all_solutions)

is_lead = st.sidebar.radio("Is Lead?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

snapshot = {
    "total_activities": total_activities,
    "unique_activities": unique_activities,
    "last_action": last_action,
    "country": country,
    "solution": solution,
    "is_lead": is_lead
}

if st.sidebar.button("Predict Outcome", type="primary"):
    
    prob = system.predict_win_probability(snapshot)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Win Probability")
        st.metric(label="Probability Score", value=f"{prob:.1%}")
        
        if prob > 0.7:
            st.success("High Probability")
        elif prob > 0.4:
            st.warning("Medium Probability")
        else:
            st.error("Low Probability")

    with col2:
        st.subheader("Recommended Next Actions")
        st.write(f"Context: {country} | {solution} | After {last_action}")
        
        next_actions = system.get_best_next_actions(country, solution, last_action)
        
        if next_actions.empty:
            st.info("No historical matches found for this specific context.")
        else:
            st.dataframe(next_actions, use_container_width=True)

    st.markdown("---")

    st.subheader("Top Successful Paths")
    best_paths = system.get_best_paths()
    st.table(best_paths)
    
    st.markdown("---")
    
    st.subheader("Market Insights")
    tab1, tab2 = st.tabs(["By Country", "By Solution"])
    
    with tab1:
        st.dataframe(system.get_top_actions(by="country", top_n=5), use_container_width=True)
    
    with tab2:
        st.dataframe(system.get_top_actions(by="solution", top_n=5), use_container_width=True)

else:
    st.info("Adjust parameters in the sidebar and click 'Predict Outcome' to start analysis.")
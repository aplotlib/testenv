import streamlit as st
import pandas as pd
from datetime import datetime
import io
import matplotlib.pyplot as plt

class ReturnRxSimple:
    def __init__(self):
        self.scenarios = pd.DataFrame(columns=[
            'scenario_name', 'sku', 'sales_30', 'avg_sale_price', 
            'sales_channel', 'returns_30', 'solution', 'solution_cost',
            'additional_cost_per_item', 'current_unit_cost', 'reduction_rate',
            'return_cost_30', 'return_cost_annual', 'revenue_impact_30',
            'revenue_impact_annual', 'new_unit_cost', 'savings_30',
            'annual_savings', 'break_even_days', 'break_even_months',
            'roi', 'score', 'timestamp'])

    def add_scenario(self, scenario_name, sku, sales_30, avg_sale_price, sales_channel, 
                     returns_30, solution, solution_cost, additional_cost_per_item, 
                     current_unit_cost, reduction_rate):
        if not scenario_name:
            scenario_name = f"Scenario {len(self.scenarios) + 1}"

        return_cost_30 = returns_30 * current_unit_cost
        return_cost_annual = return_cost_30 * 12
        revenue_impact_30 = returns_30 * avg_sale_price
        revenue_impact_annual = revenue_impact_30 * 12
        new_unit_cost = current_unit_cost + additional_cost_per_item

        savings_30 = returns_30 * (reduction_rate / 100) * avg_sale_price
        annual_savings = savings_30 * 12

        roi = None
        break_even_days = None
        break_even_months = None
        score = None

        if solution_cost > 0 and annual_savings > 0:
            roi = annual_savings / solution_cost
            annual_additional_costs = additional_cost_per_item * sales_30 * 12
            if (annual_savings - annual_additional_costs) > 0:
                break_even_days = solution_cost / (annual_savings - annual_additional_costs)
                break_even_months = break_even_days / 30
                score = roi * 100 - break_even_days

        new_row = {
            'scenario_name': scenario_name,
            'sku': sku,
            'sales_30': sales_30,
            'avg_sale_price': avg_sale_price,
            'sales_channel': sales_channel,
            'returns_30': returns_30,
            'solution': solution,
            'solution_cost': solution_cost,
            'additional_cost_per_item': additional_cost_per_item,
            'current_unit_cost': current_unit_cost,
            'reduction_rate': reduction_rate,
            'return_cost_30': return_cost_30,
            'return_cost_annual': return_cost_annual,
            'revenue_impact_30': revenue_impact_30,
            'revenue_impact_annual': revenue_impact_annual,
            'new_unit_cost': new_unit_cost,
            'savings_30': savings_30,
            'annual_savings': annual_savings,
            'break_even_days': break_even_days,
            'break_even_months': break_even_months,
            'roi': roi,
            'score': score,
            'timestamp': datetime.now()
        }

        self.scenarios = pd.concat([self.scenarios, pd.DataFrame([new_row])], ignore_index=True)


# Streamlit App UI
st.set_page_config(page_title="ReturnRx Enterprise", layout="centered")
st.title("📦 ReturnRx Enterprise - Product Return Optimizer")

# Initialize app
if "app" not in st.session_state:
    st.session_state.app = ReturnRxSimple()
app = st.session_state.app

# Input form
st.subheader("Add New Scenario")
with st.form("scenario_form"):
    scenario_name = st.text_input("Scenario Name")
    sku = st.text_input("SKU")
    sales_30 = st.number_input("Sales (30 days)", min_value=0.0)
    avg_sale_price = st.number_input("Average Sale Price", min_value=0.0)
    sales_channel = st.text_input("Top Sales Channel")
    returns_30 = st.number_input("Returns (30 days)", min_value=0.0)
    solution = st.text_input("Suggested Solution")
    solution_cost = st.number_input("Solution Total Cost", min_value=0.0)
    additional_cost_per_item = st.number_input("Additional Cost per Item", value=0.0)
    current_unit_cost = st.number_input("Current Unit Cost", min_value=0.0)
    reduction_rate = st.slider("Est. Return Rate Reduction (%)", 0, 100)

    submitted = st.form_submit_button("Add Scenario")
    if submitted:
        if sku.strip() == "":
            st.error("SKU is required.")
        else:
            app.add_scenario(
                scenario_name, sku, sales_30, avg_sale_price, sales_channel,
                returns_30, solution, solution_cost, additional_cost_per_item,
                current_unit_cost, reduction_rate
            )
            st.success(f"Scenario '{scenario_name or 'Default'}' added.")

# Show all scenarios
st.divider()
st.subheader("📊 All Scenarios")

if app.scenarios.empty:
    st.info("No scenarios added yet.")
else:
    # Filter section
    with st.expander("🔍 Filter Options"):
        selected_sku = st.selectbox("Filter by SKU", ["All"] + sorted(app.scenarios['sku'].unique().tolist()))
        selected_channel = st.selectbox("Filter by Sales Channel", ["All"] + sorted(app.scenarios['sales_channel'].unique().tolist()))
        filtered_df = app.scenarios.copy()

        if selected_sku != "All":
            filtered_df = filtered_df[filtered_df['sku'] == selected_sku]
        if selected_channel != "All":
            filtered_df = filtered_df[filtered_df['sales_channel'] == selected_channel]

    # Display filtered results
    st.dataframe(filtered_df, use_container_width=True)

    # CSV Export Button
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered CSV",
        data=csv,
        file_name="scenarios_export.csv",
        mime="text/csv"
    )

    # ROI Chart
    st.subheader("📈 ROI by Scenario")
    roi_chart_data = filtered_df.dropna(subset=['roi'])
    if not roi_chart_data.empty:
        fig, ax = plt.subplots()
        ax.bar(roi_chart_data['scenario_name'], roi_chart_data['roi'])
        ax.set_ylabel("ROI")
        ax.set_title("ROI per Scenario")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    else:
        st.info("No ROI data available to plot.")

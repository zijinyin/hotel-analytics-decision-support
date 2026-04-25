import streamlit as st
import pandas as pd
import plotly.express as px

from live_app.utils.cancellation_inference import predict_cancellation
from live_app.utils.customer_value_inference import predict_customer_value


st.set_page_config(page_title="Hotel Analytics Decision Support System", layout="wide")

# ===============================
# Light hotel-themed styling
# ===============================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF8E8;
        color: #3E3A35;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    h1, h2, h3, h4 {
        color: #4E4337;
    }

    .stCaption {
        color: #7B6D5D;
    }

    section[data-testid="stSidebar"] {
        background-color: #F7EEDB;
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    button[data-baseweb="tab"] {
        background: #F4E8C8;
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        border: 1px solid #E2D3AC;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #E9D59A;
        color: #3E3A35;
        font-weight: 600;
    }

    div.stButton > button {
        background-color: #D9B96E;
        color: #2F2A24;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #CFAA57;
        color: #2F2A24;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFDF8;
        border: 1px solid #E7D9B6;
        padding: 12px 16px;
        border-radius: 14px;
        box-shadow: 0 1px 3px rgba(60, 50, 30, 0.05);
    }

    .hotel-card {
        background-color: #FFFDF8;
        border: 1px solid #E7D9B6;
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(60, 50, 30, 0.05);
    }

    .hotel-subtle {
        color: #7B6D5D;
        font-size: 0.96rem;
    }

    div[data-testid="stExpander"] {
        background-color: #FFFDF8;
        border: 1px solid #E7D9B6;
        border-radius: 14px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .small-note {
        color: #8A7A67;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Helpers
# ===============================
def format_percent(x):
    return f"{x * 100:.1f}%"


def cancel_label_text(label: int) -> str:
    return "Likely to Cancel" if int(label) == 1 else "Not Likely to Cancel"


def customer_value_text(label: int) -> str:
    return "High Value" if int(label) == 1 else "Low Value"


def recommendation_text(cancel_prob: float, customer_label: int) -> str:
    if cancel_prob >= 0.7 and customer_label == 1:
        return "High-priority follow-up recommended. This booking has high cancellation risk and involves a high-value customer."
    elif cancel_prob >= 0.7 and customer_label == 0:
        return "Monitor closely. This booking has high cancellation risk, but the customer is lower value."
    elif cancel_prob < 0.7 and customer_label == 1:
        return "Stable booking and valuable guest. Maintain service quality and normal engagement."
    else:
        return "Low immediate concern. No urgent action is needed."


@st.cache_data
def load_sample_customers():
    return pd.read_csv("live_app/sample_data/sample_customers.csv")


def get_customer_preview(df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "Nationality",
        "Age",
        "DaysSinceCreation",
        "AverageLeadTime",
        "BookingsCheckedIn",
        "BookingsCanceled",
        "BookingsNoShowed",
        "LodgingRevenue",
        "OtherRevenue",
    ]
    keep = [c for c in preferred_cols if c in df.columns]
    if keep:
        return df[keep]
    return df.head(1)


def get_strategy_examples():
    return pd.DataFrame([
        {
            "name": "Emma Carter",
            "booking_risk": 0.86,
            "customer_value": 0.89,
            "quadrant": "Q1: High Risk + High Value",
            "booking_note": "Long lead time, no deposit, cancellation history",
            "customer_note": "Repeat high-spend guest",
            "action": "Highest priority. Contact immediately, confirm plans, and consider retention actions because this booking is both risky and strategically important."
        },
        {
            "name": "Daniel Brooks",
            "booking_risk": 0.28,
            "customer_value": 0.84,
            "quadrant": "Q2: Low Risk + High Value",
            "booking_note": "Stable reservation with stronger commitment signals",
            "customer_note": "High-value guest with strong past behavior",
            "action": "Maintain relationship. No urgent intervention is needed, but this guest should receive strong service attention and relationship management."
        },
        {
            "name": "Olivia Chen",
            "booking_risk": 0.82,
            "customer_value": 0.26,
            "quadrant": "Q3: High Risk + Low Value",
            "booking_note": "Risky booking with weaker commitment",
            "customer_note": "Lower-value guest profile",
            "action": "Monitor closely, but do not over-invest. This booking is risky, but the customer is lower value, so lower-cost operational handling is more appropriate."
        },
        {
            "name": "Liam Walker",
            "booking_risk": 0.24,
            "customer_value": 0.22,
            "quadrant": "Q4: Low Risk + Low Value",
            "booking_note": "Low-risk and stable booking",
            "customer_note": "Lower-value guest profile",
            "action": "Normal handling. This case does not require urgent attention or special resource allocation."
        },
    ])


# ===============================
# Load data
# ===============================
sample_customers = load_sample_customers()
strategy_df = get_strategy_examples()

# ===============================
# Sidebar
# ===============================
with st.sidebar.expander("Advanced Model Configuration"):
    cancel_model_path = st.text_input("Cancellation model path", value="xgb_best_model.pkl")
    cancel_feature_path = st.text_input("Cancellation feature list path", value="feature_list.pkl")
    customer_model_path = st.text_input("Customer value model path", value="xgb_customer_value_model.pkl")
    customer_feature_path = st.text_input("Customer value feature list path", value="customer_feature_list.pkl")

# ===============================
# Header
# ===============================
st.markdown(
    """
    <div class="hotel-card" style="padding-top: 26px; padding-bottom: 18px;">
        <h1 style="margin-bottom: 0.25rem;">Hotel Analytics Decision Support System</h1>
        <div class="hotel-subtle">
            A hospitality-themed prototype combining project reporting, strategic decision guidance, and live prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Project Dashboard", "Strategy Matrix", "Live Prediction"]
)

# ===============================
# TAB 1: OVERVIEW
# ===============================
with tab1:
    st.markdown('<div class="hotel-card">', unsafe_allow_html=True)
    st.subheader("Project Overview")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Business Problem")
        st.write(
            "Hotels need to identify risky bookings early and understand which customers are most valuable "
            "so they can prioritize follow-up, manage capacity more effectively, and improve operational decision-making."
        )

        st.markdown("### Current Modeling Tasks")
        st.write("- Booking cancellation prediction")
        st.write("- Customer value classification")

    with c2:
        st.markdown("### Current AWS Components")
        st.write("- Amazon S3 for datasets, scripts, and model artifacts")
        st.write("- Amazon SageMaker for training and scoring workflows")
        st.write("- Amazon RDS MySQL for prediction results and metadata")
        st.write("- EC2 + Streamlit for the live demo application")

    st.markdown("---")

    st.markdown("### System Purpose")
    st.write(
        "This prototype combines two layers: a reporting layer that summarizes data and model results, "
        "and a live prediction layer that lets users evaluate new booking scenarios with customer context."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# TAB 2: PROJECT DASHBOARD
# ===============================
with tab2:
    st.markdown('<div class="hotel-card">', unsafe_allow_html=True)
    st.subheader("Project Dashboard")
    st.caption("Key project information extracted from the two model development notebooks.")

    st.markdown("### Dataset Overview")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("#### Dataset 1 — Booking Cancellation")
        st.metric("Rows", "119,390")
        st.metric("Columns", "32")
        st.metric("Cancellation Rate", "37.04%")
        st.markdown(
            """
            **Cleaning & preprocessing**
            - Removed 180 invalid rows with 0 guests
            - Dropped leakage / excluded columns such as reservation status and assigned room type
            - Key missingness handled in `children`, `agent`, and `company`
            - Final training matrix: **119,210 × 34**
            """
        )

    with d2:
        st.markdown("#### Dataset 2 — Customer Value")
        st.metric("Raw Rows", "83,590")
        st.metric("Active Customers", "63,670")
        st.metric("Value Split", "50% / 50%")
        st.markdown(
            """
            **Cleaning & preprocessing**
            - Filtered to customers with at least 1 check-in
            - Dropped anonymized ID/hash columns
            - Age cleaned and imputed
            - Target defined using median lodging revenue: **€302.27**
            - Final feature matrix: **63,670 × 31**
            """
        )

    st.markdown("---")

    st.markdown("### Feature Engineering Highlights")

    f1, f2 = st.columns(2)

    with f1:
        st.markdown("#### Booking Cancellation Features")
        st.write(
            """
            - `total_nights`
            - `total_guests`
            - `has_agent`
            - `has_company`
            - `has_special_requests`
            - `lead_time_log`
            - `adr_log`
            - `agent_log`
            - `company_log`
            - `cancel_rate_history`
            """
        )

    with f2:
        st.markdown("#### Customer Value Features")
        st.write(
            """
            - `total_special_requests`
            - `has_canceled`
            - `has_noshowed`
            - `is_repeat`
            - `tenure_days`
            - `AverageLeadTime_log`
            - `DaysSinceCreation_log`
            """
        )

    st.markdown("---")

    st.markdown("### Model Performance Summary")

    booking_model_df = pd.DataFrame({
        "Model": [
            "Majority Class",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost (paper replica)",
            "XGBoost (tuned)"
        ],
        "F1": [0.0000, 0.7082, 0.7267, 0.8126, 0.7722, 0.7887],
        "AUC": [0.5000, 0.8661, 0.8790, 0.9338, 0.9170, 0.9255]
    })

    customer_model_df = pd.DataFrame({
        "Model": [
            "Majority Class",
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "F1": [0.0000, 0.7119, 0.7704, 0.7540],
        "AUC": [0.5000, 0.7779, 0.8549, 0.8223]
    })

    m1, m2 = st.columns(2)

    with m1:
        st.markdown("#### Booking Cancellation Model Comparison")
        st.dataframe(booking_model_df, use_container_width=True)
        st.markdown("**Best model shown in dashboard:** Random Forest")
        st.write("- Test F1: **0.8126**")
        st.write("- Test AUC: **0.9338**")

    with m2:
        st.markdown("#### Customer Value Model Comparison")
        st.dataframe(customer_model_df, use_container_width=True)
        st.markdown("**Best model shown in dashboard:** Random Forest")
        st.write("- Test F1: **0.7704**")
        st.write("- Test AUC: **0.8549**")

    st.markdown("---")

    st.markdown("### Final Model Selection")

    s1, s2 = st.columns(2)

    with s1:
        st.success(
            "Best model shown in dashboard: **Random Forest for booking cancellation prediction**"
        )
        st.write(
            "For dashboard reporting, Random Forest is presented as the strongest booking-cancellation model "
            "because it achieved the highest test F1 and AUC among the compared models."
        )

    with s2:
        st.success(
            "Best model shown in dashboard: **Random Forest for customer value classification**"
        )
        st.write(
            "For dashboard reporting, Random Forest is presented as the strongest customer-value model "
            "because it achieved the highest test F1 and AUC among the compared models."
        )

    st.markdown("---")

    st.markdown("### Current AWS System Status")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Storage", "Amazon S3")
    a2.metric("ML Environment", "SageMaker")
    a3.metric("Database", "RDS MySQL")
    a4.metric("Live App", "EC2 + Streamlit")

    st.info(
        "The current prototype stores data and model artifacts in S3, uses SageMaker for training and scoring, "
        "stores structured outputs in RDS, and runs the live demo app on EC2 using Streamlit."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# TAB 3: STRATEGY MATRIX
# ===============================
with tab3:
    st.markdown('<div class="hotel-card">', unsafe_allow_html=True)
    st.subheader("Strategy Matrix")
    st.caption(
        "This page illustrates how booking risk and customer value can be combined into simple operational strategy zones."
    )

    st.markdown("### Four Action Zones")

    z1, z2, z3, z4 = st.columns(4)

    card_style = """
        background-color: #FFFDF8;
        border: 1px solid #E7D9B6;
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 92px;
    """

    label_style = "font-size: 0.95rem; color: #6F6356; margin-bottom: 6px;"
    value_style = "font-size: 1.35rem; font-weight: 600; color: #3E3A35; line-height: 1.2;"

    with z1:
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="{label_style}">Q1</div>
                <div style="{value_style}">High Risk + High Value</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with z2:
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="{label_style}">Q2</div>
                <div style="{value_style}">Low Risk + High Value</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with z3:
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="{label_style}">Q3</div>
                <div style="{value_style}">High Risk + Low Value</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with z4:
        st.markdown(
            f"""
            <div style="{card_style}">
                <div style="{label_style}">Q4</div>
                <div style="{value_style}">Low Risk + Low Value</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    fig = px.scatter(
        strategy_df,
        x="customer_value",
        y="booking_risk",
        color="quadrant",
        hover_name="name",
        hover_data={
            "customer_value": ':.2f',
            "booking_risk": ':.2f',
            "booking_note": True,
            "customer_note": True,
            "quadrant": True
        },
        size=[18, 18, 18, 18],
        size_max=18
    )

    fig.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="#8B7D6B")
    fig.add_hline(y=0.5, line_width=2, line_dash="dash", line_color="#8B7D6B")

    fig.update_layout(
        paper_bgcolor="#FFFDF8",
        plot_bgcolor="#FFFDF8",
        xaxis_title="Customer Value",
        yaxis_title="Booking Risk",
        legend_title="Quadrant",
        margin=dict(l=20, r=20, t=30, b=20)
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="small-note">Hover over a point to preview the simulated guest or booking scenario.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Select a Simulated Case")

    selected_name = st.selectbox(
        "Choose a persona from the matrix",
        strategy_df["name"].tolist()
    )

    selected_row = strategy_df[strategy_df["name"] == selected_name].iloc[0]

    info1, info2 = st.columns([1, 1])

    with info1:
        st.markdown("#### Example Case")
        st.write(f"**Name:** {selected_row['name']}")
        st.write(f"**Quadrant:** {selected_row['quadrant']}")
        st.write(f"**Booking Risk Score:** {selected_row['booking_risk']:.2f}")
        st.write(f"**Customer Value Score:** {selected_row['customer_value']:.2f}")
        st.write(f"**Booking Context:** {selected_row['booking_note']}")
        st.write(f"**Customer Context:** {selected_row['customer_note']}")

    with info2:
        st.markdown("#### Recommended Handling")
        st.success(selected_row["action"])

        if selected_row["quadrant"].startswith("Q1"):
            st.write("Reason: this is the most important group because the booking looks unstable and the customer is strategically valuable.")
        elif selected_row["quadrant"].startswith("Q2"):
            st.write("Reason: the booking is stable, so the main priority shifts from intervention to relationship maintenance.")
        elif selected_row["quadrant"].startswith("Q3"):
            st.write("Reason: the booking is risky, but the customer is lower value, so lighter-touch operational action is usually more efficient.")
        else:
            st.write("Reason: both booking risk and customer value are relatively low, so standard workflow is usually enough.")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# TAB 4: LIVE PREDICTION
# ===============================
with tab4:
    st.markdown('<div class="hotel-card">', unsafe_allow_html=True)
    st.subheader("1. Booking Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        hotel = st.selectbox("Hotel", ["Resort Hotel", "City Hotel"])
        lead_time = st.number_input("Lead Time", min_value=0, value=100)
        arrival_date_month = st.selectbox(
            "Arrival Month",
            [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
        )
        arrival_date_week_number = st.number_input("Arrival Week Number", min_value=1, max_value=53, value=27)
        arrival_date_day_of_month = st.number_input("Arrival Day of Month", min_value=1, max_value=31, value=15)
        stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
        stays_in_week_nights = st.number_input("Week Nights", min_value=0, value=2)
        adults = st.number_input("Adults", min_value=0, value=2)
        children = st.number_input("Children", min_value=0, value=0)
        babies = st.number_input("Babies", min_value=0, value=0)

    with col2:
        meal = st.selectbox("Meal", ["BB", "HB", "FB", "SC"])
        market_segment = st.selectbox(
            "Market Segment",
            ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary", "Aviation"]
        )
        distribution_channel = st.selectbox(
            "Distribution Channel",
            ["TA/TO", "Direct", "Corporate", "GDS"]
        )
        is_repeated_guest = st.selectbox("Repeated Guest", [0, 1])
        previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
        previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)
        reserved_room_type = st.selectbox("Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])

    with col3:
        deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Refundable", "Non Refund"])
        customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
        agent = st.number_input("Agent", min_value=0, value=0)
        company = st.number_input("Company", min_value=0, value=0)
        adr = st.number_input("ADR", min_value=0.0, value=120.0, step=1.0)
        required_car_parking_spaces = st.number_input("Parking Spaces", min_value=0, value=1)
        total_of_special_requests = st.number_input("Special Requests", min_value=0, value=1)

    booking_input = pd.DataFrame([{
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "agent": agent,
        "company": company,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
    }])

    st.subheader("2. Customer Profile")

    sample_customers_display = sample_customers.copy().reset_index(drop=True)
    sample_customers_display["profile_label"] = "Customer " + (sample_customers_display.index + 1).astype(str)

    selected_customer_label = st.selectbox(
        "Select a sample customer profile",
        sample_customers_display["profile_label"].tolist()
    )

    selected_idx = sample_customers_display[
        sample_customers_display["profile_label"] == selected_customer_label
    ].index[0]

    selected_customer = sample_customers_display.loc[[selected_idx]].drop(columns=["profile_label"])
    preview_customer = get_customer_preview(selected_customer)

    with st.expander("Preview selected customer profile", expanded=True):
        st.dataframe(preview_customer, use_container_width=True)

    st.subheader("3. Run Live Prediction")

    if st.button("Run Prediction"):
        try:
            cancel_result = predict_cancellation(
                input_df=booking_input,
                model_path=cancel_model_path,
                feature_list_path=cancel_feature_path
            )

            customer_result = predict_customer_value(
                input_df=selected_customer,
                model_path=customer_model_path,
                feature_list_path=customer_feature_path
            )

            cancel_label = int(cancel_result["prediction_label"].iloc[0])
            cancel_prob = float(cancel_result["prediction_probability"].iloc[0])
            risk_level = str(cancel_result["risk_level"].iloc[0])

            customer_label = int(customer_result["prediction_label"].iloc[0])
            customer_prob = float(customer_result["prediction_probability"].iloc[0])
            value_level = str(customer_result["value_level"].iloc[0])

            recommendation = recommendation_text(cancel_prob, customer_label)

            st.subheader("4. Prediction Results")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown("### Cancellation Risk")
                st.metric("Prediction", cancel_label_text(cancel_label))
                st.metric("Probability", format_percent(cancel_prob))
                st.metric("Risk Level", risk_level)

            with r2:
                st.markdown("### Customer Value")
                st.metric("Prediction", customer_value_text(customer_label))
                st.metric("Probability", format_percent(customer_prob))
                st.metric("Value Level", value_level)

            with r3:
                st.markdown("### Decision Support")
                st.success(recommendation)

            with st.expander("Booking input used", expanded=False):
                st.dataframe(booking_input, use_container_width=True)

            with st.expander("Customer profile used", expanded=False):
                st.dataframe(preview_customer, use_container_width=True)

        except Exception as e:
            st.error(f"Error while running prediction: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

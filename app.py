import io
import pandas as pd
import streamlit as st
import sys
import os
from datetime import datetime
import tempfile
import shutil

# Add the current directory to Python path to import our custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the cons date processor module
from cons_date_processor import extract_cons_numbers, get_commit_date_from_csv, map_cons_to_commit_date

# Initialize model_loaded flag before any other operations
model_loaded = False
predictor = None

try:
    from train_rollover_model import RolloverPredictor
    predictor = RolloverPredictor()
    model_loaded = True
except ImportError:
    st.sidebar.error("Cannot import RolloverPredictor. Make sure train_rollover_model.py is in the same directory.")
    model_loaded = False
except Exception as e:
    st.sidebar.error(f"Error loading ML model: {str(e)}")
    model_loaded = False

# Initialize session state for cons numbers and commit dates at the top level
if 'extracted_cons_numbers' not in st.session_state:
    st.session_state.extracted_cons_numbers = []
if 'cons_commit_dates' not in st.session_state:
    st.session_state.cons_commit_dates = {}
if 'uploaded_csv_files' not in st.session_state:
    st.session_state.uploaded_csv_files = {}

################################################################################
#                               WIDGET LAYOUT                                  #
################################################################################
st.set_page_config(page_title="FedEx Backlog Recovery System", page_icon="📦", layout="wide")

# Header with FedEx branding colors
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4B0082 0%, #FF4500 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4B0082;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4B0082;
        margin: 0;
    }
    .metric-label {
        color: #666;
        margin: 0;
        font-size: 0.9rem;
    }
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    .custom-uld-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #4B0082;
        margin-bottom: 1rem;
    }
    .uld-priority-tag {
        background: #4B0082;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.3rem;
    }
    .capacity-section {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #28a745;
        margin-bottom: 1rem;
    }
    .capacity-metric {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.3rem;
    }
    .commit-date-section {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #007bff;
        margin-bottom: 1rem;
    }
    .commit-date-tag {
        background: #007bff;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class=\"main-header\"><h1>🚀 FedEx Backlog Recovery System with Capacity Handling & Commit Date Priority</h1></div>", unsafe_allow_html=True)

# Main file upload
uploaded_file = st.file_uploader(
    "📂 Upload Backlog Summary Report (.xlsx in standard format)", 
    type="xlsx",
    help="Select your daily backlog Excel file with 'Backlog Details' sheet"
)

# NEW: Reporting Ramp Selection
st.sidebar.subheader("📍 Reporting Ramp Selection")
reporting_ramp_options = ["All Ramps", "BLR", "DWC", "BOM", "DEL"]
selected_reporting_ramp = st.sidebar.selectbox(
    "Select Reporting Ramp",
    options=reporting_ramp_options,
    index=0,
    help="Filter the analysis by a specific reporting ramp"
)

# NEW: Commit Date Priority Section
st.markdown("<div class=\"commit-date-section\">", unsafe_allow_html=True)
st.markdown("### 📅 Commit Date Priority (NEW)")
st.markdown("**Prioritize shipments based on commit dates from uploaded Cons/AWB files**")

# Enable/Disable commit date priority
use_commit_date_priority = st.checkbox(
    "📅 Enable Commit Date Priority", 
    value=False,
    help="When enabled, shipments will be prioritized based on commit dates from uploaded CSV files"
)

if use_commit_date_priority:
    # Extract Cons/AWB numbers from uploaded backlog file
    if uploaded_file:
        try:
            # Save uploaded file temporarily to extract cons numbers
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract cons numbers
            cons_numbers = extract_cons_numbers(tmp_file_path)
            st.session_state.extracted_cons_numbers = cons_numbers
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            if cons_numbers:
                st.success(f"✅ Found {len(cons_numbers)} 12-digit Cons/AWB numbers in the backlog file")
                
                # Display first few cons numbers
                st.markdown("**Sample Cons/AWB Numbers found:**")
                sample_cons = cons_numbers[:5] if len(cons_numbers) > 5 else cons_numbers
                for i, cons in enumerate(sample_cons, 1):
                    st.write(f"{i}. {cons}")
                if len(cons_numbers) > 5:
                    st.write(f"... and {len(cons_numbers) - 5} more")
                
                # File uploader for CSV files
                st.markdown("**Upload CSV files for Cons/AWB numbers:**")
                uploaded_csv_files = st.file_uploader(
                    "Upload CSV files containing commit date information",
                    type="csv",
                    accept_multiple_files=True,
                    help="Upload CSV files corresponding to the Cons/AWB numbers found above"
                )
                
                if uploaded_csv_files:
                    # Process uploaded CSV files
                    processed_files = {}
                    cons_commit_mapping = {}
                    
                    for csv_file in uploaded_csv_files:
                        # Save CSV file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                            tmp_csv.write(csv_file.getvalue())
                            tmp_csv_path = tmp_csv.name
                        
                        # Extract cons number and commit date from CSV
                        found_cons, commit_date = get_commit_date_from_csv(tmp_csv_path)
                        
                        if found_cons and commit_date:
                            # Check if this cons number is in our extracted list
                            if found_cons in cons_numbers:
                                cons_commit_mapping[found_cons] = commit_date
                                processed_files[found_cons] = csv_file.name
                            else:
                                st.warning(f"⚠️ Cons number {found_cons} from {csv_file.name} not found in backlog report")
                        else:
                            st.error(f"❌ Could not extract commit date from {csv_file.name}")
                        
                        # Clean up temporary file
                        os.unlink(tmp_csv_path)
                    
                    # Update session state
                    st.session_state.cons_commit_dates = cons_commit_mapping
                    st.session_state.uploaded_csv_files = processed_files
                    
                    # Display results
                    if cons_commit_mapping:
                        st.success(f"✅ Successfully processed {len(cons_commit_mapping)} CSV files")
                        
                        # Show mapping table
                        st.markdown("**Cons/AWB to Commit Date Mapping:**")
                        mapping_df = pd.DataFrame([
                            {"Cons/AWB Number": cons, "Commit Date": date, "Source File": processed_files.get(cons, "Unknown")}
                            for cons, date in cons_commit_mapping.items()
                        ])
                        st.dataframe(mapping_df, use_container_width=True)
                    else:
                        st.warning("⚠️ No valid commit dates could be extracted from uploaded CSV files")
            else:
                st.warning("⚠️ No 12-digit Cons/AWB numbers found in the backlog file")
        except Exception as e:
            st.error(f"❌ Error processing backlog file: {str(e)}")
    else:
        st.info("📋 Please upload a backlog file first to extract Cons/AWB numbers")

st.markdown("</div>", unsafe_allow_html=True)

# NEW: Capacity Handling Section
st.markdown("<div class=\"capacity-section\">", unsafe_allow_html=True)
st.markdown("### ✈️ Flight Capacity Configuration")
st.markdown("**Configure flight capacity and planned shipments for capacity-based rollover calculation**")

# Enable/Disable capacity-based rollover
use_capacity_rollover = st.checkbox(
    "🛫 Enable Capacity-Based Rollover", 
    value=False,
    help="When enabled, rollover will be calculated based on flight capacity constraints"
)

if use_capacity_rollover:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        flight_capacity = st.number_input(
            "Flight Capacity (kilo pounds)",
            min_value=0.0,
            value=100.0,
            step=1.0,
            help="Total capacity of the flight in kilo pounds"
        )
    
    with col2:
        planned_shipments = st.number_input(
            "Planned Shipments Weight (kilo pounds)",
            min_value=0.0,
            value=50.0,
            step=1.0,
            help="Weight of shipments already planned for this flight"
        )
    
    with col3:
        # Unit selection
        weight_unit = st.selectbox(
            "Weight Unit",
            options=["kilo pounds", "pounds"],
            index=0,
            help="Select the unit for capacity and weight calculations"
        )

st.markdown("</div>", unsafe_allow_html=True)

# NEW: Custom ULD Priority Section
st.markdown("<div class=\"custom-uld-section\">", unsafe_allow_html=True)
st.markdown("### 🎯 Custom ULD Priority (Optional)")
st.markdown("**Add specific ULD numbers to prioritize at the top of the list**")

# Initialize session state for custom ULDs if not exists
if 'custom_ulds' not in st.session_state:
    st.session_state.custom_ulds = []

# Enable/Disable custom ULD priority
use_custom_uld_priority = st.checkbox(
    "🔥 Enable Custom ULD Priority", 
    value=False,
    help="When enabled, specified ULD numbers will be prioritized at the top"
)

if use_custom_uld_priority:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input for ULD number
        new_uld = st.text_input(
            "Enter ULD Number:",
            placeholder="e.g., PMC61834FX",
            help="Type ULD number and click 'Add' to include in priority list"
        )
    
    with col2:
        st.write("")  # Empty space for alignment
        if st.button("➕ Add ULD", disabled=not new_uld.strip()):
            if new_uld.strip() and new_uld.strip() not in st.session_state.custom_ulds:
                st.session_state.custom_ulds.append(new_uld.strip())
                st.success(f"Added {new_uld.strip()} to priority list!")
                st.rerun()
            elif new_uld.strip() in st.session_state.custom_ulds:
                st.warning("ULD already in priority list!")
    
    # Display current priority ULDs
    if st.session_state.custom_ulds:
        st.markdown("**Current Priority ULDs (in order):**")
        for i, uld in enumerate(st.session_state.custom_ulds, 1):
            col1, col2 = st.columns([4, 1])
            with col1:
                # Corrected f-string syntax
                st.markdown(f'<span class="uld-priority-tag">#{i}</span> {uld}', unsafe_allow_html=True)
            with col2:
                if st.button(f"🗑️", key=f"remove_{uld}", help=f"Remove {uld}"):
                    st.session_state.custom_ulds.remove(uld)
                    st.rerun()
        
        # Clear all button
        if st.button("🗑️ Clear All Priority ULDs"):
            st.session_state.custom_ulds = []
            st.rerun()
    else:
        st.info("No custom priority ULDs added yet.")

st.markdown("</div>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration Panel")

# Default weightings – user can override in the sidebar
st.sidebar.subheader("🎯 Priority Code Weights")
default_order = ["IBX", "IPF", "IEB", "IXF"]
weights = {}
for i, code in enumerate(default_order, start=1):
    weights[code] = st.sidebar.number_input(
        f"{code} (rank value – lower is better)", 
        min_value=1,
        value=i, 
        key=f"weight_{code}",
        help=f"Set priority ranking for {code} shipments"
    )

use_build_date = st.sidebar.checkbox(
    "📅 Use ULD Build Date as tie-breaker (earlier ⇒ higher)", 
    value=True,
    help="When two shipments have same priority, prioritize older build dates"
)

# === NEW HYBRID PRIORITIZATION CONTROLS ===
st.sidebar.subheader("🚨 Hybrid Age-Based Priority")

age_threshold = st.sidebar.number_input(
    "Priority Age Threshold (Days)",
    min_value=0, max_value=30, value=3,
    help="Shipments older than this will get priority boost",
    key="age_threshold"
)

priority_boost_categories = st.sidebar.multiselect(
    "Boost These Categories When Aged",
    options=["IBX", "IPF", "IEB", "IXF"],
    default=["IBX", "IEB"],
    help="Selected categories will jump ahead when exceeding age threshold",
    key="boost_categories"
)

use_hybrid_priority = st.sidebar.checkbox(
    "Enable Hybrid Age-Based Priority", 
    value=True,
    help="Use age + category hybrid prioritization",
    key="hybrid_enabled"
)

# Rollover method selection
st.sidebar.subheader("🤖 Rollover Prediction Method")

# Add capacity-based rollover option
rollover_options = ["Manual Input", "ML Prediction"]
if use_capacity_rollover:
    rollover_options.append("Capacity-Based Rollover")

rollover_method = st.sidebar.radio(
    "Choose rollover prediction method:",
    rollover_options,
    help="Manual: Set percentage manually | ML: Use machine learning prediction | Capacity: Use flight capacity constraints"
)

# Initialize rollover percentage variable
roll_pct = 10  # default value

if rollover_method == "Manual Input":
    roll_pct = st.sidebar.slider(
        "🎚️ Roll-over percentage (bottom X %)", 
        min_value=0, 
        max_value=100, 
        value=10,
        help="Percentage of lowest priority shipments to roll over to next day"
    )
    st.sidebar.info("ℹ️ Using manual rollover percentage")
elif rollover_method == "ML Prediction":
    if model_loaded:
        st.sidebar.success("✅ ML model loaded successfully")
        st.sidebar.info("🧠 Will use AI prediction when file is processed")
    else:
        st.sidebar.error("❌ ML model not available")
        st.sidebar.warning("⚠️ Falling back to manual input")
        roll_pct = st.sidebar.slider(
            "🎚️ Roll-over percentage (bottom X %)", 
            min_value=0, 
            max_value=100, 
            value=10
        )
elif rollover_method == "Capacity-Based Rollover":
    st.sidebar.success("✈️ Using capacity-based rollover")
    st.sidebar.info("🎯 Rollover will be calculated based on flight capacity constraints")

################################################################################
#                               FEATURE EXTRACTION FOR ML                      #
################################################################################
def extract_features_for_ml(df):
    """Extract features from backlog DataFrame for ML prediction"""
    features = {}
    
    # Basic metrics
    features["total_ulds"] = len(df)
    features["total_net_weight"] = df["Net Weight (LBS)"].sum()
    features["total_gross_weight"] = df["Gross Weight (LBS)"].sum()
    features["avg_weight_per_uld"] = df["Net Weight (LBS)"].mean()
    
    # Priority distribution
    priority_counts = df["Priority"].value_counts(normalize=True)
    for code in ["IBX", "IEB", "IPF", "IXF", "IEF"]:
        features[f"pct_{code.lower()}"] = priority_counts.get(code, 0)
    
    # Position distribution
    if "ULD Position" in df.columns:
        pos_counts = df["ULD Position"].value_counts(normalize=True)
        for pos in ["MD", "LD", "LW"]:
            features[f"pct_{pos.lower()}"] = pos_counts.get(pos, 0)
    else:
        for pos in ["MD", "LD", "LW"]:
            features[f"pct_{pos.lower()}"] = 0
    
    # Region distribution
    if "ULD Destination Region" in df.columns:
        reg_counts = df["ULD Destination Region"].value_counts(normalize=True)
        for reg in ["AM", "EU", "MEISA", "AS"]:
            features[f"pct_{reg.lower()}"] = reg_counts.get(reg, 0)
    else:
        for reg in ["AM", "EU", "MEISA", "AS"]:
            features[f"pct_{reg.lower()}"] = 0
    
    # Ramp distribution
    if "Reporting Ramp" in df.columns:
        ramp_counts = df["Reporting Ramp"].value_counts(normalize=True)
        for ramp in ["DWC", "BOM", "DEL", "BLR"]:
            features[f"pct_{ramp.lower()}"] = ramp_counts.get(ramp, 0)
    else:
        for ramp in ["DWC", "BOM", "DEL", "BLR"]:
            features[f"pct_{ramp.lower()}"] = 0
    
    # Reason distribution
    reason_col = None
    for col_name in ["Reason for Backlog", "Backlog Reason"]:
        if col_name in df.columns:
            reason_col = col_name
            break
    
    if reason_col:
        reason_counts = df[reason_col].value_counts(normalize=True)
        features["pct_no_ops"] = reason_counts.get("No Ops/No Network available", 0)
        features["pct_demand_over"] = reason_counts.get("Demand over allocation", 0)
        features["pct_space_constraint"] = reason_counts.get("Space constraint/No space for uplift", 0)
        features["pct_planned_rollover"] = reason_counts.get("Planned rollover/As per capacity plan", 0)
    else:
        features["pct_no_ops"] = 0
        features["pct_demand_over"] = 0
        features["pct_space_constraint"] = 0
        features["pct_planned_rollover"] = 0
    
    # Time features
    try:
        date = pd.to_datetime(df["Backlog Reporting Date"].iloc[0])
        features["day_of_week"] = date.dayofweek
        features["day_of_month"] = date.day
        features["week_of_year"] = date.isocalendar()[1]
    except:
        features["day_of_week"] = 0
        features["day_of_month"] = 1
        features["week_of_year"] = 1
    
    return pd.DataFrame([features])

################################################################################
#                               CORE LOGIC                                     #
################################################################################
def prioritize(df, weights, use_build_date, age_threshold=0, priority_boost_categories=[], use_hybrid_priority=False, custom_ulds=[], use_custom_uld_priority=False, cons_commit_dates={}, use_commit_date_priority=False):
    """Enhanced prioritization with hybrid age-based priority boost, custom ULD priority, and commit date priority"""
    
    # Calculate shipment age in days
    current_date = datetime.now()
    df["ULD Build Date"] = pd.to_datetime(df["ULD Build Date"])
    df["Age_Days"] = (current_date - df["ULD Build Date"]).dt.days
    
    # NEW: Apply commit date priority
    if use_commit_date_priority and cons_commit_dates:
        # Add commit date information to the dataframe
        df["Commit Date"] = df["CONS/AWB Number"].apply(
            lambda cons: cons_commit_dates.get(str(int(cons)) if pd.notna(cons) else "", None)
        )
        
        # Create commit date rank (earlier dates get lower rank numbers = higher priority)
        df["Commit_Date_Rank"] = df["Commit Date"].apply(
            lambda date: date.toordinal() if pd.notna(date) else 999999
        )
        
        # Add visual indicator for commit date priority
        df["Commit Date Priority"] = df["Commit Date"].apply(
            lambda date: f"📅 {date}" if pd.notna(date) else "No Date"
        )
    else:
        df["Commit_Date_Rank"] = 999999
        df["Commit Date Priority"] = "No Date"
        df["Commit Date"] = None
    
    # NEW: Apply custom ULD priority
    if use_custom_uld_priority and custom_ulds:
        # Create custom priority rank (0 for custom ULDs, 1 for others)
        df["Custom_ULD_Rank"] = df.apply(
            lambda row: custom_ulds.index(row["ULD Number"]) if pd.notna(row["ULD Number"]) and row["ULD Number"] in custom_ulds else 999,
            axis=1
        )
        df["Is_Custom_Priority"] = df["Custom_ULD_Rank"] < 999
        
        # Add visual indicator for custom priority ULDs
        df["Custom Priority"] = df.apply(
            lambda row: f"🔥 CUSTOM #{custom_ulds.index(row['ULD Number']) + 1}" if row["Is_Custom_Priority"] else "Normal",
            axis=1
        )
    else:
        df["Custom_ULD_Rank"] = 999
        df["Is_Custom_Priority"] = False
        df["Custom Priority"] = "Normal"
    
    # Apply hybrid priority boost logic
    if use_hybrid_priority and age_threshold > 0 and priority_boost_categories:
        df["Priority Boost"] = df.apply(
            lambda row: "⭐ AGED" if (row["Age_Days"] >= age_threshold and 
                                     row["Priority"] in priority_boost_categories) else "Normal",
            axis=1
        )
        # Create boost rank (0 for boosted, 1 for normal)
        df["Boost_Rank"] = df["Priority Boost"].apply(lambda x: 0 if x == "⭐ AGED" else 1)
    else:
        df["Priority Boost"] = "Normal"
        df["Boost_Rank"] = 1
    
    # Map priority codes to numeric ranks
    df["Code_Rank"] = df["Priority"].map(weights)
    
    # BuildDate handling
    if use_build_date:
        df["Build_Rank"] = df["ULD Build Date"]
    else:
        df["Build_Rank"] = pd.Timestamp.max  # neutral
    
    # Multi-tier sorting: Custom_ULD_Rank (custom ULDs first), Commit_Date_Rank (earlier dates first), Boost_Rank (0=boosted first), Code_Rank, Build_Rank
    df = df.sort_values(["Custom_ULD_Rank", "Commit_Date_Rank", "Boost_Rank", "Code_Rank", "Build_Rank"]).reset_index(drop=True)
    df["Priority Rank"] = df.index + 1
    
    # Clean up helper columns but keep Priority Boost, Age_Days, Custom Priority, and Commit Date Priority for display
    df = df.drop(columns=["Code_Rank", "Build_Rank", "Boost_Rank", "Custom_ULD_Rank", "Is_Custom_Priority", "Commit_Date_Rank"])
    return df


def split_rollover(df, pct):
    """Original percentage-based rollover split"""
    if pct == 0:
        return df, pd.DataFrame(columns=df.columns)
    cutoff = int(len(df) * (100 - pct) / 100)
    cleared = df.iloc[:cutoff].copy()
    rollover = df.iloc[cutoff:].copy()
    return cleared, rollover


def split_by_weight_rollover(df, target_rollover_weight):
    """NEW: Weight-based rollover split for capacity handling"""
    if target_rollover_weight <= 0:
        return df, pd.DataFrame(columns=df.columns)
    
    # Sort by priority rank (lowest priority first for rollover)
    df_sorted = df.sort_values("Priority Rank", ascending=False).reset_index(drop=True)
    
    # Store original indices before reset_index
    df_sorted["original_index"] = df.sort_values("Priority Rank", ascending=False).index
    
    # Calculate cumulative weight from lowest priority
    df_sorted["Cumulative_Weight"] = df_sorted["Net Weight (LBS)"].cumsum()
    
    # Find the cutoff point where cumulative weight exceeds target rollover weight
    rollover_mask = df_sorted["Cumulative_Weight"] <= target_rollover_weight
    
    # If no ULDs meet the exact weight, include the first ULD that exceeds it
    if not rollover_mask.any():
        # No rollover needed if target is very small
        return df, pd.DataFrame(columns=df.columns)
    elif rollover_mask.all():
        # All ULDs need to be rolled over
        return pd.DataFrame(columns=df.columns), df
    else:
        # Find the last ULD that should be rolled over
        last_rollover_idx = rollover_mask.sum()
        
        # Check if we need to include one more ULD to meet the target weight
        if last_rollover_idx < len(df_sorted):
            remaining_weight = target_rollover_weight - df_sorted.iloc[last_rollover_idx - 1]["Cumulative_Weight"]
            if remaining_weight > 0:
                # Include the next ULD to better meet the target
                last_rollover_idx += 1
        
        rollover_ulds = df_sorted.iloc[:last_rollover_idx]
        cleared_ulds = df_sorted.iloc[last_rollover_idx:]
    
    # Get the original dataframe indices for the rollover and cleared ULDs
    rollover_original_indices = rollover_ulds["original_index"].tolist()
    cleared_original_indices = cleared_ulds["original_index"].tolist()
    
    # Return subsets of original dataframe maintaining original order
    rollover = df.loc[rollover_original_indices].sort_values("Priority Rank")
    cleared = df.loc[cleared_original_indices].sort_values("Priority Rank")
    
    return cleared, rollover


def calculate_capacity_rollover(total_backlog_weight, flight_capacity, planned_shipments, weight_unit="kilo pounds"):
    """Calculate rollover weight based on capacity constraints"""
    # Convert to consistent units (pounds)
    conversion_factor = 1000 if weight_unit == "kilo pounds" else 1
    
    flight_capacity_lbs = flight_capacity * conversion_factor
    planned_shipments_lbs = planned_shipments * conversion_factor
    
    # Calculate total demand
    total_demand_lbs = planned_shipments_lbs + total_backlog_weight
    
    # Calculate rollover weight
    rollover_weight_lbs = max(0, total_demand_lbs - flight_capacity_lbs)
    
    return rollover_weight_lbs, flight_capacity_lbs, planned_shipments_lbs, total_demand_lbs


def show_deck_wise_data(df):
    """Display deck-wise distribution of ULDs"""
    if "ULD Position" not in df.columns:
        return
    
    st.markdown("### 🏗️ Deck-wise Distribution")
    
    pos_summary = df.groupby("ULD Position").agg({
        "ULD Number": "count",
        "Net Weight (LBS)": "sum"
    }).round(0)
    
    col1, col2, col3 = st.columns(3)
    
    positions = ["MD", "LD", "LW"]
    colors = ["🟠", "🟣", "🔵"]
    
    for i, (pos, color) in enumerate(zip(positions, colors)):
        with [col1, col2, col3][i]:
            if pos in pos_summary.index:
                count = int(pos_summary.loc[pos, "ULD Number"])
                weight = int(pos_summary.loc[pos, "Net Weight (LBS)"])
            else:
                count = 0
                weight = 0
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{color} {pos} Position</h4>
                <p><strong>{count}</strong> ULDs</p>
                <p><strong>{weight:,}</strong> lbs</p>
            </div>
            """, unsafe_allow_html=True)

################################################################################
#                               MAIN WORKFLOW                                  #
################################################################################
if uploaded_file:
    try:
        raw = pd.read_excel(uploaded_file, sheet_name="Backlog Details")
        
        # Validate required columns
        required_cols = ["Priority", "ULD Build Date", "Net Weight (LBS)"]
        if selected_reporting_ramp != "All Ramps":
            required_cols.append("Reporting Ramp")

        missing_cols = [col for col in required_cols if col not in raw.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()
            
        # Filter by Reporting Ramp if selected
        if selected_reporting_ramp != "All Ramps":
            raw = raw[raw["Reporting Ramp"] == selected_reporting_ramp].copy()
            if raw.empty:
                st.warning(f"⚠️ No ULDs found for Reporting Ramp: {selected_reporting_ramp}")
                st.stop()
            else:
                st.info(f"✅ Displaying analysis for Reporting Ramp: {selected_reporting_ramp}")

        # Check if ULD Number column exists for custom priority feature
        if use_custom_uld_priority and "ULD Number" not in raw.columns:
            st.error("❌ 'ULD Number' column is required for custom ULD priority feature")
            st.stop()
            
        # Check if CONS/AWB Number column exists for commit date priority feature
        if use_commit_date_priority and "CONS/AWB Number" not in raw.columns:
            st.error("❌ 'CONS/AWB Number' column is required for commit date priority feature")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Could not read 'Backlog Details' sheet: {e}")
        st.stop()
    
    # Calculate total backlog weight
    total_backlog_weight = raw["Net Weight (LBS)"].sum()
    
    # Rollover calculation based on method
    if rollover_method == "Capacity-Based Rollover" and use_capacity_rollover:
        # Calculate capacity-based rollover
        rollover_weight_lbs, flight_capacity_lbs, planned_shipments_lbs, total_demand_lbs = calculate_capacity_rollover(
            total_backlog_weight, flight_capacity, planned_shipments, weight_unit
        )
        
        # Display capacity calculation
        st.info(f"✈️ **Capacity Analysis**: Flight capacity: {flight_capacity_lbs:,.0f} lbs | "
                f"Planned shipments: {planned_shipments_lbs:,.0f} lbs | "
                f"Total backlog: {total_backlog_weight:,.0f} lbs | "
                f"**Rollover required**: {rollover_weight_lbs:,.0f} lbs")
        
        use_weight_based_rollover = True
        target_rollover_weight = rollover_weight_lbs
        
    elif rollover_method == "ML Prediction" and model_loaded:
        try:
            # Extract features for ML prediction
            features_df = extract_features_for_ml(raw)
            
            # Make prediction
            predicted_pct = predictor.predict(features_df)[0]
            
            # Ensure prediction is within reasonable bounds
            predicted_pct = max(0, min(100, predicted_pct))
            
            # Use the predicted percentage
            roll_pct = predicted_pct
            use_weight_based_rollover = False
            
            # Display prediction info
            st.info(f"🧠 **ML Prediction**: {predicted_pct:.1f}% rollover recommended")
            
        except Exception as e:
            st.warning(f"⚠️ ML prediction failed: {str(e)}")
            st.info("📝 Using default 10% rollover instead")
            roll_pct = 10
            use_weight_based_rollover = False
    else:
        # Manual input
        use_weight_based_rollover = False
    
    # Run prioritization with custom ULD priority and commit date priority
    prioritized = prioritize(
        raw, weights, use_build_date, age_threshold, priority_boost_categories, 
        use_hybrid_priority, st.session_state.custom_ulds, use_custom_uld_priority,
        st.session_state.cons_commit_dates, use_commit_date_priority
    )

    # Split for roll-over based on method
    if rollover_method == "Capacity-Based Rollover" and use_capacity_rollover:
        cleared, rollover = split_by_weight_rollover(prioritized, target_rollover_weight)
    else:
        cleared, rollover = split_rollover(prioritized, roll_pct)
    
    # Calculate weight totals
    cleared_wt = cleared["Net Weight (LBS)"].sum()
    rollover_wt = rollover["Net Weight (LBS)"].sum()
    total_wt = prioritized["Net Weight (LBS)"].sum()
    
    # Prepare in-memory workbooks
    def to_bytes(df_dict):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            for name, frame in df_dict.items():
                frame.to_excel(xw, sheet_name=name, index=False)
        buf.seek(0)
        return buf
    
    # Display results summary
    st.markdown("## 📊 Processing Results")
    
    # Show capacity analysis if applicable
    if rollover_method == "Capacity-Based Rollover" and use_capacity_rollover:
        st.markdown("### ✈️ Flight Capacity Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <p class="metric-value">{flight_capacity_lbs:,.0f}</p>
                <p class="metric-label">Flight Capacity (lbs)</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <p class="metric-value">{planned_shipments_lbs:,.0f}</p>
                <p class="metric-label">Planned Shipments (lbs)</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <p class="metric-value">{total_demand_lbs:,.0f}</p>
                <p class="metric-label">Total Demand (lbs)</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            utilization_pct = (min(total_demand_lbs, flight_capacity_lbs) / flight_capacity_lbs) * 100
            st.markdown(f'''
            <div class="metric-card">
                <p class="metric-value">{utilization_pct:.1f}%</p>
                <p class="metric-label">Capacity Utilization</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Show commit date priority status
    if use_commit_date_priority and st.session_state.cons_commit_dates:
        commit_date_ulds = prioritized[prioritized["Commit Date Priority"] != "No Date"]
        if not commit_date_ulds.empty:
            st.success(f"📅 {len(commit_date_ulds)} ULD(s) prioritized by commit date")
        else:
            st.warning("⚠️ No ULDs found with commit date information")
    
    # Show custom ULD priority status
    if use_custom_uld_priority and st.session_state.custom_ulds:
        custom_ulds_found = prioritized[prioritized["Custom Priority"] != "Normal"]["ULD Number"].tolist()
        custom_ulds_not_found = [uld for uld in st.session_state.custom_ulds if uld not in custom_ulds_found]
        
        if custom_ulds_found:
            st.success(f"✅ {len(custom_ulds_found)} custom priority ULD(s) found and prioritized: {', '.join(custom_ulds_found)}")
        
        if custom_ulds_not_found:
            st.warning(f"⚠️ {len(custom_ulds_not_found)} custom priority ULD(s) not found in data: {', '.join(custom_ulds_not_found)}")
    
    # Create metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-value">{len(prioritized)}</p>
            <p class="metric-label">Total ULDs</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-value">{total_wt:,.0f}</p>
            <p class="metric-label">Total Weight (lbs)</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-value">{len(cleared)}</p>
            <p class="metric-label">To Clear Today</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-value">{len(rollover)}</p>
            <p class="metric-label">Rolled Over</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Weight summary
    st.markdown("### ⚖️ Weight Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="✅ Net weight to clear", 
            value=f"{cleared_wt:,.0f} lbs",
            delta=f"{100 * cleared_wt / total_wt:.1f}% of total"
        )
    
    with col2:
        st.metric(
            label="⏭️ Net weight rolled over", 
            value=f"{rollover_wt:,.0f} lbs",
            delta=f"{100 * rollover_wt / total_wt:.1f}% of total"
        )
    
    # Display priority breakdown
    st.markdown("### 📋 Priority Breakdown")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Priority Summary", "📅 Commit Date Priority", "🔥 Custom Priority ULDs", "📈 Age Analysis", "✈️ Capacity Details"])
    
    with tab1:
        priority_summary = prioritized["Priority"].value_counts().sort_index()
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Priority Code Distribution:**")
            for code, count in priority_summary.items():
                pct = 100 * count / len(prioritized)
                st.write(f"• **{code}**: {count} ULDs ({pct:.1f}%)")
        
        with col2:
            if use_hybrid_priority:
                boost_summary = prioritized["Priority Boost"].value_counts()
                st.markdown("**Age-Based Priority Boost:**")
                for status, count in boost_summary.items():
                    pct = 100 * count / len(prioritized)
                    st.write(f"• **{status}**: {count} ULDs ({pct:.1f}%)")
    
    with tab2:
        if use_commit_date_priority and st.session_state.cons_commit_dates:
            commit_date_ulds = prioritized[prioritized["Commit Date Priority"] != "No Date"]
            if not commit_date_ulds.empty:
                st.markdown("**ULDs Prioritized by Commit Date:**")
                display_cols = ["Priority Rank", "ULD Number", "Commit Date Priority", "Priority", "Net Weight (LBS)", "ULD Build Date"]
                available_cols = [col for col in display_cols if col in commit_date_ulds.columns]
                st.dataframe(commit_date_ulds[available_cols], use_container_width=True)
            else:
                st.info("No ULDs found with commit date information.")
        else:
            st.info("Commit date priority is not enabled or no commit dates available.")
    
    with tab3:
        if use_custom_uld_priority and st.session_state.custom_ulds:
            custom_priority_ulds = prioritized[prioritized["Custom Priority"] != "Normal"]
            if not custom_priority_ulds.empty:
                st.markdown("**Custom Priority ULDs (Top Priority):**")
                display_cols = ["Priority Rank", "ULD Number", "Custom Priority", "Priority", "Net Weight (LBS)", "ULD Build Date"]
                available_cols = [col for col in display_cols if col in custom_priority_ulds.columns]
                st.dataframe(custom_priority_ulds[available_cols], use_container_width=True)
            else:
                st.info("No custom priority ULDs found in the current data.")
        else:
            st.info("Custom ULD priority is not enabled or no ULDs specified.")
    
    with tab4:
        if "Age_Days" in prioritized.columns:
            avg_age = prioritized["Age_Days"].mean()
            max_age = prioritized["Age_Days"].max()
            min_age = prioritized["Age_Days"].min()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Age", f"{avg_age:.1f} days")
            with col2:
                st.metric("Oldest Shipment", f"{max_age} days")
            with col3:
                st.metric("Newest Shipment", f"{min_age} days")
            
            if use_hybrid_priority and age_threshold > 0:
                aged_count = len(prioritized[prioritized["Age_Days"] >= age_threshold])
                st.info(f"📅 {aged_count} ULDs are {age_threshold}+ days old and eligible for age-based priority boost")
    
    with tab5:
        if rollover_method == "Capacity-Based Rollover" and use_capacity_rollover:
            st.markdown("**Capacity Calculation Details:**")
            
            capacity_data = {
                "Parameter": [
                    "Flight Capacity",
                    "Planned Shipments",
                    "Current Backlog",
                    "Total Demand",
                    "Available Capacity",
                    "Rollover Required"
                ],
                "Value (lbs)": [
                    f"{flight_capacity_lbs:,.0f}",
                    f"{planned_shipments_lbs:,.0f}",
                    f"{total_backlog_weight:,.0f}",
                    f"{total_demand_lbs:,.0f}",
                    f"{max(0, flight_capacity_lbs - planned_shipments_lbs):,.0f}",
                    f"{rollover_weight_lbs:,.0f}"
                ],
                "Description": [
                    "Maximum weight capacity of the flight",
                    "Weight of shipments already planned for this flight",
                    "Total weight of current backlog",
                    "Planned shipments + Current backlog",
                    "Remaining capacity after planned shipments",
                    "Weight that must be rolled over due to capacity constraints"
                ]
            }
            
            capacity_df = pd.DataFrame(capacity_data)
            st.table(capacity_df)
            
            # Show formula
            st.markdown("**Calculation Formula:**")
            st.code("""
            Total Demand = Planned Shipments + Current Backlog
            Rollover Required = max(0, Total Demand - Flight Capacity)
            """)
        else:
            st.info("Capacity-based rollover is not enabled for this calculation.")
    
    # Display deck-wise data
    show_deck_wise_data(raw)
    
    # Data preview
    st.markdown("### 👀 Data Preview")
    
    # Show top 10 prioritized items
    preview_cols = ["Priority Rank", "ULD Number", "Commit Date Priority", "Custom Priority", "Priority Boost", "Priority", "Net Weight (LBS)", "ULD Build Date", "Age_Days"]
    available_preview_cols = [col for col in preview_cols if col in prioritized.columns]
    st.dataframe(prioritized.head(10)[available_preview_cols], use_container_width=True)
    
    # Download buttons
    st.markdown("### 📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Full prioritized list
        full_bytes = to_bytes({"Prioritized_Backlog": prioritized})
        st.download_button(
            label="📋 Download Full Prioritized List",
            data=full_bytes,
            file_name=f"Prioritized_Backlog_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # Items to clear today
        clear_bytes = to_bytes({"To_Clear_Today": cleared})
        st.download_button(
            label="✅ Download Items to Clear",
            data=clear_bytes,
            file_name=f"To_Clear_Today_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        # Rollover items
        if not rollover.empty:
            rollover_bytes = to_bytes({"Rollover_Items": rollover})
            st.download_button(
                label="⏭️ Download Rollover Items",
                data=rollover_bytes,
                file_name=f"Rollover_Items_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("No rollover items")
    
    # Combined download
    st.markdown("---")
    combined_bytes = to_bytes({
        "Prioritized_Backlog": prioritized,
        "To_Clear_Today": cleared,
        "Rollover_Items": rollover
    })
    
    st.download_button(
        label="📦 Download Complete Package (All Sheets)",
        data=combined_bytes,
        file_name=f"Backlog_Recovery_Package_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

else:
    # Instructions when no file is uploaded
    st.markdown("## 📖 How to Use")
    
    st.markdown("""
    ### 🚀 Getting Started
    1. **Upload** your daily backlog Excel file using the file uploader above
    2. **Configure** priority weights and settings in the sidebar
    3. **Select Reporting Ramp** (NEW) to filter analysis by specific ramp
    4. **Enable Commit Date Priority** (NEW) to prioritize by commit dates from CSV files
    5. **Add Custom ULD Priority** (optional) to prioritize specific ULD numbers
    6. **Configure Flight Capacity** (optional) for capacity-based rollover calculation
    7. **Choose** rollover prediction method (Manual, ML, or Capacity-Based)
    8. **Download** your prioritized results
    
    ### 📅 NEW: Commit Date Priority Feature
    - **Enable** commit date priority to use commit dates from uploaded CSV files
    - **Automatic extraction** of 12-digit Cons/AWB numbers from backlog report
    - **Upload multiple CSV files** containing commit date information
    - **Automatic mapping** of Cons/AWB numbers to commit dates
    - **Earlier commit dates** get higher priority in the final ranking
    
    ### ✈️ Capacity-Based Rollover
    - **Enable** capacity-based rollover to use flight capacity constraints
    - **Input** flight capacity and planned shipments weight
    - **Automatic calculation** of rollover based on available capacity
    - **More accurate** rollover predictions based on operational constraints
    
    ### 🎯 Custom ULD Priority Feature
    - **Enable** the custom ULD priority option above
    - **Add** specific ULD numbers that need immediate priority
    - **Order matters**: ULDs are prioritized in the order you add them
    - **Visual indicators** show which ULDs received custom priority
    
    ### 🤖 Rollover Methods
    - **Manual Input**: Set rollover percentage manually based on experience
    - **ML Prediction**: Use machine learning to predict optimal rollover percentage
    - **Capacity-Based**: Calculate rollover based on flight capacity constraints
    
    ### 📊 Priority System (Updated)
    - **Commit Date Priority**: Earliest commit dates get highest priority (NEW!)
    - **Custom ULD Priority**: Override system priority for specific ULDs
    - **IBX, IPF, IEB, IXF**: Configure priority weights (lower number = higher priority)
    - **Age-Based Boost**: Automatically boost priority for aged shipments
    - **Build Date Tie-breaker**: Use ULD build date to break ties
    
    ### 📥 Export Options
    - **Prioritized List**: Complete ranked list of all ULDs with commit date info
    - **Items to Clear**: Shipments scheduled for today
    - **Rollover Items**: Shipments deferred to next day
    - **Complete Package**: All sheets in one Excel file
    """)
    
    # Sample data format
    st.markdown("### 📋 Required Data Format")
    st.markdown("""
    Your Excel file should contain a sheet named **'Backlog Details'** with these columns:
    - **Priority**: Priority code (IBX, IPF, IEB, IXF)
    - **ULD Build Date**: Build date of the ULD
    - **Net Weight (LBS)**: Net weight in pounds
    - **ULD Number**: Unique ULD identifier (required for custom priority)
    - **CONS/AWB Number**: Cons/AWB numbers (required for commit date priority)
    - **Reporting Ramp**: The reporting ramp for the ULD (required if filtering by ramp)
    - **ULD Position**: Position type (MD, LD, LW) - optional
    - **Other columns**: Additional data will be preserved in output
    
    **CSV files for commit dates should contain:**
    - **Cons/AWB number** in cells A4 or A5
    - **Commit dates** in column K ('Cmit Date')
    - **Standard CSV format** with proper headers
    """)






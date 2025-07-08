
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import os
# import glob

# def calculate_rollover_from_consecutive_days(folder_path):
#     """
#     Calculate rollover percentages by matching ULDs between consecutive days
#     """
#     # Get all Excel files in the folder
#     excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
#     excel_files.sort()  # Sort by filename to get chronological order

#     rollover_data = []

#     for i in range(len(excel_files) - 1):
#         current_file = excel_files[i]
#         next_file = excel_files[i + 1]

#         try:
#             # Read current day and next day files
#             current_df = pd.read_excel(current_file, sheet_name="Backlog Details")
#             next_df = pd.read_excel(next_file, sheet_name="Backlog Details")

#             # Create composite keys for matching (ULD Number + Net Weight + Origin Ramp)
#             current_df['composite_key'] = (current_df['ULD Number'].astype(str) + "_" + 
#                                          current_df['Net Weight (LBS)'].astype(str) + "_" + 
#                                          current_df['ULD Origin Ramp'].astype(str))

#             next_df['composite_key'] = (next_df['ULD Number'].astype(str) + "_" + 
#                                       next_df['Net Weight (LBS)'].astype(str) + "_" + 
#                                       next_df['ULD Origin Ramp'].astype(str))

#             # Find ULDs that appear in both days (rolled over)
#             rolled_over_keys = set(current_df['composite_key']).intersection(set(next_df['composite_key']))
#             rolled_over_df = current_df[current_df['composite_key'].isin(rolled_over_keys)]

#             # Calculate rollover percentages
#             total_ulds = len(current_df)
#             rolled_over_ulds = len(rolled_over_df)
#             rollover_count_pct = (rolled_over_ulds / total_ulds) * 100 if total_ulds > 0 else 0

#             total_weight = current_df['Net Weight (LBS)'].sum()
#             rolled_over_weight = rolled_over_df['Net Weight (LBS)'].sum()
#             rollover_weight_pct = (rolled_over_weight / total_weight) * 100 if total_weight > 0 else 0

#             # Extract features from current day
#             current_date = os.path.basename(current_file).replace('.xlsx', '')

#             # Basic backlog profile features
#             total_net_weight = current_df['Net Weight (LBS)'].sum()
#             total_gross_weight = current_df['Gross Weight (LBS)'].sum()
#             avg_weight_per_uld = total_net_weight / total_ulds if total_ulds > 0 else 0

#             # Priority distribution
#             priority_counts = current_df['Priority'].value_counts()
#             ibx_pct = (priority_counts.get('IBX', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#             ipf_pct = (priority_counts.get('IPF', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#             ieb_pct = (priority_counts.get('IEB', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#             ixf_pct = (priority_counts.get('IXF', 0) / total_ulds) * 100 if total_ulds > 0 else 0

#             # Position distribution
#             position_counts = current_df['ULD Position'].value_counts()
#             md_pct = (position_counts.get('MD', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#             ld_pct = (position_counts.get('LD', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#             lw_pct = (position_counts.get('LW', 0) / total_ulds) * 100 if total_ulds > 0 else 0

#             # Region distribution
#             origin_region_counts = current_df['Origin Region'].value_counts()
#             dest_region_counts = current_df['ULD Destination Region'].value_counts()

#             # Reason code analysis
#             reason_counts = current_df['Reason for Backlog'].value_counts()
#             no_ops_pct = (reason_counts.get('No Ops/No network available', 0) / total_ulds) * 100 if total_ulds > 0 else 0
#             demand_over_alloc_pct = (reason_counts.get('Demand over allocation', 0) / total_ulds) * 100 if total_ulds > 0 else 0

#             # Time-based features
#             try:
#                 date_obj = datetime.strptime(current_date.split('_')[-1] if '_' in current_date else current_date, '%Y%m%d')
#                 day_of_week = date_obj.weekday()  # 0=Monday, 6=Sunday
#                 week_of_year = date_obj.isocalendar()[1]
#             except:
#                 day_of_week = 0
#                 week_of_year = 1

#             # Store the data
#             rollover_record = {
#                 'date': current_date,
#                 'total_ulds': total_ulds,
#                 'rolled_over_ulds': rolled_over_ulds,
#                 'rollover_count_pct': rollover_count_pct,
#                 'rollover_weight_pct': rollover_weight_pct,
#                 'total_net_weight': total_net_weight,
#                 'total_gross_weight': total_gross_weight,
#                 'avg_weight_per_uld': avg_weight_per_uld,
#                 'ibx_pct': ibx_pct,
#                 'ipf_pct': ipf_pct,
#                 'ieb_pct': ieb_pct,
#                 'ixf_pct': ixf_pct,
#                 'md_pct': md_pct,
#                 'ld_pct': ld_pct,
#                 'lw_pct': lw_pct,
#                 'no_ops_pct': no_ops_pct,
#                 'demand_over_alloc_pct': demand_over_alloc_pct,
#                 'day_of_week': day_of_week,
#                 'week_of_year': week_of_year,
#                 'dominant_origin_region': origin_region_counts.index[0] if len(origin_region_counts) > 0 else 'Unknown',
#                 'dominant_dest_region': dest_region_counts.index[0] if len(dest_region_counts) > 0 else 'Unknown'
#             }

#             rollover_data.append(rollover_record)

#             print(f"Processed: {current_date} -> {os.path.basename(next_file).replace('.xlsx', '')}")
#             print(f"  Rollover: {rolled_over_ulds}/{total_ulds} ULDs ({rollover_count_pct:.1f}%), {rolled_over_weight:.0f}/{total_weight:.0f} lbs ({rollover_weight_pct:.1f}%)")

#         except Exception as e:
#             print(f"Error processing {current_file} -> {next_file}: {str(e)}")

#     # Convert to DataFrame
#     rollover_df = pd.DataFrame(rollover_data)
#     return rollover_df

# if __name__ == "__main__":
#     # Example usage
#     folder_path = "backlog_files"  # Folder containing Excel files
#     rollover_df = calculate_rollover_from_consecutive_days(folder_path)

#     # Save the training data
#     rollover_df.to_csv("rollover_training_data.csv", index=False)
#     print(f"\nTraining data saved with {len(rollover_df)} samples")
#     print("\nSample features:")
#     print(rollover_df.head())
#==========================================================================================================
#Version 2

import pandas as pd
import numpy as np
import glob
import os

def load_backlog_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name="Backlog Details")
        # Ensure all necessary columns exist
        for col in ["ULD Number", "Net Weight (LBS)", "Gross Weight (LBS)", "ULD Origin Ramp"]:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_composite_key(df):
    # Fill missing values to avoid NaN in key
    df['ULD Number'] = df['ULD Number'].fillna('').astype(str)
    df['Net Weight (LBS)'] = df['Net Weight (LBS)'].fillna(0).astype(str)
    df['ULD Origin Ramp'] = df['ULD Origin Ramp'].fillna('').astype(str)
    df['Composite_Key'] = (
        df['ULD Number'] + '_' +
        df['Net Weight (LBS)'] + '_' +
        df['ULD Origin Ramp']
    )
    return df

def calculate_rollover_features(df):
    features = {}
    features['total_ulds'] = len(df)
    features['total_net_weight'] = df['Net Weight (LBS)'].sum()
    features['total_gross_weight'] = df['Gross Weight (LBS)'].sum()
    features['avg_weight_per_uld'] = df['Net Weight (LBS)'].mean()
    # Priority distribution
    priority_counts = df['Priority'].value_counts(normalize=True)
    for code in ['IBX', 'IEB', 'IPF', 'IXF', 'IEF']:
        features[f'pct_{code.lower()}'] = priority_counts.get(code, 0)
    # Position
    pos_counts = df['ULD Position'].value_counts(normalize=True)
    for pos in ['MD', 'LD', 'LW']:
        features[f'pct_{pos.lower()}'] = pos_counts.get(pos, 0)
    # Region
    reg_counts = df['ULD Destination Region'].value_counts(normalize=True)
    for reg in ['AM', 'EU', 'MEISA', 'AS']:
        features[f'pct_{reg.lower()}'] = reg_counts.get(reg, 0)
    # Ramp
    ramp_counts = df['Reporting Ramp'].value_counts(normalize=True)
    for ramp in ['DWC', 'BOM', 'DEL', 'BLR']:
        features[f'pct_{ramp.lower()}'] = ramp_counts.get(ramp, 0)
    # Reason
    reason_col = "Backlog Reason" if "Backlog Reason" in df.columns else "Reason for Backlog"
    reason_counts = df[reason_col].value_counts(normalize=True)
    features['pct_no_ops'] = reason_counts.get('No Ops/No Network available', 0)
    features['pct_demand_over'] = reason_counts.get('Demand over allocation', 0)
    features['pct_space_constraint'] = reason_counts.get('Space constraint/No space for uplift', 0)
    features['pct_planned_rollover'] = reason_counts.get('Planned rollover/As per capacity plan', 0)
    # Time
    try:
        date = pd.to_datetime(df['Backlog Reporting Date'].iloc[0])
        features['day_of_week'] = date.dayofweek
        features['day_of_month'] = date.day
        features['week_of_year'] = date.isocalendar()[1]
    except Exception:
        features['day_of_week'] = features['day_of_month'] = features['week_of_year'] = -1
    return features

def find_rollover_ulds(day_t_df, day_t1_df):
    day_t_keys = set(day_t_df['Composite_Key'])
    day_t1_keys = set(day_t1_df['Composite_Key'])
    rollover_keys = day_t_keys & day_t1_keys
    rollover_ulds = day_t_df[day_t_df['Composite_Key'].isin(rollover_keys)]
    return rollover_ulds

def process_consecutive_days():
    training_data = []
    files = sorted(glob.glob("2025-06-*_MEISA_Backlog_Summary_Report.xlsx"))
    # Change this folder as of now.
    print(f"Found {len(files)} backlog files")
    for i in range(len(files) - 1):
        day_t_file = files[i]
        day_t1_file = files[i + 1]
        print(f"\nProcessing pair: {os.path.basename(day_t_file)} -> {os.path.basename(day_t1_file)}")
        day_t_df = load_backlog_data(day_t_file)
        day_t1_df = load_backlog_data(day_t1_file)
        if day_t_df is None or day_t1_df is None or len(day_t_df) == 0:
            print("  Skipping due to missing or empty data.")
            continue
        day_t_df = create_composite_key(day_t_df)
        day_t1_df = create_composite_key(day_t1_df)


        # Restore numeric columns for calculations
        day_t_df['Net Weight (LBS)'] = pd.to_numeric(day_t_df['Net Weight (LBS)'], errors='coerce').fillna(0)
        day_t_df['Gross Weight (LBS)'] = pd.to_numeric(day_t_df['Gross Weight (LBS)'], errors='coerce').fillna(0)
        day_t1_df['Net Weight (LBS)'] = pd.to_numeric(day_t1_df['Net Weight (LBS)'], errors='coerce').fillna(0)
        day_t1_df['Gross Weight (LBS)'] = pd.to_numeric(day_t1_df['Gross Weight (LBS)'], errors='coerce').fillna(0)

        rollover_ulds = find_rollover_ulds(day_t_df, day_t1_df)
        rollover_ulds['Net Weight (LBS)'] = pd.to_numeric(rollover_ulds['Net Weight (LBS)'], errors='coerce').fillna(0)

        total_net_weight = day_t_df['Net Weight (LBS)'].sum()
        rollover_net_weight = rollover_ulds['Net Weight (LBS)'].sum()
        count_rollover_rate = 100 * len(rollover_ulds) / len(day_t_df) if len(day_t_df) > 0 else 0
        weight_rollover_rate = 100 * rollover_net_weight / total_net_weight if total_net_weight > 0 else 0

        print(f"  Rollover: {len(rollover_ulds)}/{len(day_t_df)} ULDs ({count_rollover_rate:.1f}%)")
        print(f"  Weight rollover: {rollover_net_weight:,.0f}/{total_net_weight:,.0f} lbs ({weight_rollover_rate:.1f}%)")
        features = calculate_rollover_features(day_t_df)
        features['date'] = os.path.basename(day_t_file)
        features['count_rollover_rate'] = count_rollover_rate
        features['weight_rollover_rate'] = weight_rollover_rate
        features['cleared_count'] = len(day_t_df) - len(rollover_ulds)
        features['cleared_weight'] = total_net_weight - rollover_net_weight
        training_data.append(features)
        # train the data over which it has been generated by the name rollover_training_data
    return pd.DataFrame(training_data)

if __name__ == "__main__":
    print("=== FedEx Backlog Rollover Analysis ===")
    print("Analyzing historical rollover patterns...\n")
    training_df = process_consecutive_days()
    if len(training_df) > 0:
        training_df.to_csv('rollover_training_data.csv', index=False)
        print(f"\n✅ Training data saved with {len(training_df)} samples")
        print(f"Average rollover rate: {training_df['weight_rollover_rate'].mean():.1f}%")
        print(f"Rollover rate range: {training_df['weight_rollover_rate'].min():.1f}% - {training_df['weight_rollover_rate'].max():.1f}%")
        print("\nTraining Data Summary:")
        print(training_df[['date', 'total_ulds', 'total_net_weight', 'weight_rollover_rate']].to_string(index=False))
    else:
        print("❌ No training data generated")
 
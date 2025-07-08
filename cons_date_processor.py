import pandas as pd
import re
from collections import Counter
import io

def extract_cons_numbers(excel_file_path):
    """
    Extracts 12-digit Cons/AWB numbers from the 'CONS/AWB Number' column
    of the 'Backlog Details' sheet in the given Excel file.
    """
    try:
        df = pd.read_excel(excel_file_path, sheet_name='Backlog Details')
        cons_numbers = []
        if 'CONS/AWB Number' in df.columns:
            for cons in df['CONS/AWB Number']:
                if pd.isna(cons):
                    continue
                if isinstance(cons, (int, float)):
                    cons_str = str(int(cons))
                else:
                    cons_str = str(cons)
                
                digits_only = re.sub(r'\D', '', cons_str)
                if len(digits_only) == 12:
                    cons_numbers.append(digits_only)
        return cons_numbers
    except Exception as e:
        print(f"Error extracting Cons/AWB numbers: {e}")
        return []

def get_commit_date_from_csv(csv_file_path, target_cons_number=None):
    """
    Reads a CSV file, extracts the commit date from column 'Cmit Date' (column K),
    and returns the mode (most frequent) date for a given Cons Number.
    It dynamically finds the Cons Number and the data header.
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        found_cons = None
        header_row_index = -1

        for i, line in enumerate(lines):
            # Find Cons Number
            if 'Cons Number' in line:
                # The actual cons number is in the next line
                if i + 1 < len(lines):
                    next_line = lines[i+1]
                    potential_cons = next_line.split(',')[0]
                    digits_only = re.sub(r'\D', '', potential_cons)
                    if len(digits_only) == 12:
                        found_cons = digits_only
            
            # Find header row
            if 'Track Number' in line:
                header_row_index = i
                break

        if not found_cons:
            print(f"Error: Could not find a 12-digit Cons Number in {csv_file_path}")
            return None, None

        if header_row_index == -1:
            print(f"Error: 'Track Number' header not found in {csv_file_path}.")
            return None, None

        # Read the data part of the CSV
        # Use StringIO to treat the relevant part of the file as a new file
        csv_data = io.StringIO(''.join(lines[header_row_index:]))
        df = pd.read_csv(csv_data)

        if 'Cmit Date' in df.columns:
            # Clean the 'Cmit Date' column by removing '="' and '"'
            df['Cmit Date'] = df['Cmit Date'].astype(str).str.replace('=', '').str.replace('"', '')
            df['Cmit Date'] = pd.to_datetime(df['Cmit Date'], errors='coerce', format='%m/%d/%Y')
            valid_dates = df['Cmit Date'].dropna()
            if not valid_dates.empty:
                most_common_date = Counter(valid_dates.dt.date).most_common(1)
                if most_common_date:
                    return found_cons, most_common_date[0][0]
        
        return found_cons, None

    except Exception as e:
        print(f"Error processing CSV file {csv_file_path}: {e}")
        return None, None

def map_cons_to_commit_date(cons_numbers, csv_files_paths):
    """
    Maps Cons/AWB numbers to their commit dates by processing corresponding CSV files.
    """
    cons_commit_dates = {}
    for cons_number, file_path in csv_files_paths.items():
        found_cons, commit_date = get_commit_date_from_csv(file_path, target_cons_number=cons_number)
        if commit_date:
            cons_commit_dates[found_cons] = commit_date
    return cons_commit_dates

if __name__ == '__main__':
    # Example usage (for testing purposes)
    excel_path = '/home/ubuntu/upload/2025-07-03_MEISA_Backlog_Summary_Report.xlsx'
    csv_path = '/home/ubuntu/upload/070720250424-DESTSRV.csv'

    print("Extracting Cons/AWB numbers...")
    cons_list = extract_cons_numbers(excel_path)
    print(f"Found {len(cons_list)} 12-digit Cons/AWB numbers.")

    print(f"\nGetting commit date from {csv_path}...")
    found_cons, commit_date = get_commit_date_from_csv(csv_path)
    if commit_date:
        print(f"Commit date for {found_cons}: {commit_date}")
    else:
        print(f"Could not find commit date for {found_cons}")



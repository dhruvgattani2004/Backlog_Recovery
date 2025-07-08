import pandas as pd

excel_file = '/home/ubuntu/upload/2025-07-03_MEISA_Backlog_Summary_Report.xlsx'

try:
    df = pd.read_excel(excel_file, sheet_name='Backlog Details')
    if 'CONS/AWB Number' in df.columns:
        print("\nFirst 10 values of 'CONS/AWB Number' column:")
        print(df['CONS/AWB Number'].head(10))
        print("\nData types of 'CONS/AWB Number' column:")
        print(df['CONS/AWB Number'].dtype)
    else:
        print("\n'CONS/AWB Number' column not found.")
except Exception as e:
    print(f"Error reading Excel file: {e}")



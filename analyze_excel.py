"""
Script to analyze Excel file structure
"""
import pandas as pd
import sys
import json

def analyze_excel_file(file_path):
    """Analyze Excel file and return structure information"""
    try:
        # Read Excel file
        xl = pd.ExcelFile(file_path)

        result = {
            "file": file_path,
            "sheets": {}
        }

        # Analyze each sheet
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            sheet_info = {
                "columns": list(df.columns),
                "row_count": len(df),
                "sample_data": []
            }

            # Get 2-3 sample rows
            sample_count = min(3, len(df))
            for i in range(sample_count):
                row_data = {}
                for col in df.columns:
                    value = df.iloc[i][col]
                    # Convert to string for JSON serialization
                    if pd.isna(value):
                        row_data[col] = None
                    else:
                        row_data[col] = str(value)
                sheet_info["sample_data"].append(row_data)

            result["sheets"][sheet_name] = sheet_info

        return result

    except Exception as e:
        return {"error": str(e), "file": file_path}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_excel.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = analyze_excel_file(file_path)
    print(json.dumps(result, indent=2))

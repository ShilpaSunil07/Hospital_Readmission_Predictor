# dataset_inspector.py
# Script to inspect the actual dataset structure and fix model export

import pandas as pd
import numpy as np

def inspect_dataset():
    """Inspect the actual dataset structure"""
    print("DATASET STRUCTURE INSPECTION")
    print("=" * 50)
    
    # Load the dataset
    try:
        df = pd.read_csv('data/demo/processed_features.csv')
        print(f" Dataset loaded successfully: {df.shape}")
        
        print(f"\n DATASET INFO:")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        
        print(f"\n AVAILABLE COLUMNS:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print(f"\n SAMPLE DATA (first 3 rows):")
        print(df.head(3))
        
        print(f"\n TARGET VARIABLE:")
        if 'readmitted_within_30_days' in df.columns:
            target_counts = df['readmitted_within_30_days'].value_counts()
            print(f" Found target variable: {target_counts.to_dict()}")
        else:
            print(" Target variable 'readmitted_within_30_days' not found")
            print("Available columns that might be target:")
            for col in df.columns:
                if 'readmit' in col.lower() or 'target' in col.lower():
                    print(f"  - {col}")
        
        print(f"\n COMORBIDITY COLUMNS:")
        comorbidity_cols = []
        for col in df.columns:
            if any(term in col.lower() for term in ['diabetes', 'heart', 'renal', 'liver', 'cancer']):
                comorbidity_cols.append(col)
                print(f"   {col}")
        
        if not comorbidity_cols:
            print("   No obvious comorbidity columns found")
        
        print(f"\n NUMERIC COLUMNS:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            print(f"  - {col}: min={df[col].min()}, max={df[col].max()}")
        
        print(f"\n MISSING VALUES:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            for col, count in missing.items():
                print(f"  - {col}: {count} missing values")
        else:
            print("   No missing values found")
            
        return df
        
    except FileNotFoundError:
        print(" File not found: data/demo/processed_features.csv")
        return None
    except Exception as e:
        print(f" Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    df = inspect_dataset()
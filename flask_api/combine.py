#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import os

def combine(uuid):
    # Get all CSV files in the current directory
    os.chdir('data')
    os.chdir(uuid)
    csv_files = glob.glob('*.csv')
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file}")

    # Read and combine all CSV files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        # Extract car number plate from filename (remove .csv extension and spaces)
        car_plate = file.replace('.csv', '').replace(' ', '')
        df['CarNumberPlate'] = car_plate
        dfs.append(df)
        print(f"Loaded {file}: {len(df)} rows -> CarNumberPlate: {car_plate}")

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"\nTotal rows in combined dataset: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")


    # ## Data Cleaning - Check Missing Values

    # In[2]:


    # Check for missing values
    print("=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)

    missing_summary = pd.DataFrame({
        'Column': combined_df.columns,
        'Missing_Count': combined_df.isnull().sum(),
        'Missing_Percentage': (combined_df.isnull().sum() / len(combined_df) * 100).round(2)
    })

    print(f"\nMissing values per column:")
    print(missing_summary[missing_summary['Missing_Count'] > 0].to_string(index=False))

    # Identify columns with high percentage of missing values
    high_missing_threshold = 50  # 50% threshold
    high_missing_cols = missing_summary[missing_summary['Missing_Percentage'] > high_missing_threshold]['Column'].tolist()

    if high_missing_cols:
        print(f"\n  Columns with >{high_missing_threshold}% missing values (will be dropped):")
        for col in high_missing_cols:
            pct = missing_summary[missing_summary['Column'] == col]['Missing_Percentage'].values[0]
            print(f"  - {col}: {pct}%")
    else:
        print(f"\n✓ No columns with >{high_missing_threshold}% missing values")


    # ## Data Cleaning - Drop Columns and Duplicates

    # In[3]:


    # Drop columns with >50% missing values
    initial_shape = combined_df.shape
    print("=" * 60)
    print("DATA CLEANING STEPS")
    print("=" * 60)

    if high_missing_cols:
        combined_df = combined_df.drop(columns=high_missing_cols)
        print(f"\n✓ Dropped {len(high_missing_cols)} columns with >{high_missing_threshold}% missing values")
    else:
        print(f"\n✓ No columns dropped (no columns with >{high_missing_threshold}% missing)")

    # Check for duplicate rows
    duplicates_count = combined_df.duplicated().sum()
    print(f"\n📊 Duplicate rows found: {duplicates_count:,}")

    if duplicates_count > 0:
        # Drop duplicates
        combined_df = combined_df.drop_duplicates()
        print(f"✓ Dropped {duplicates_count:,} duplicate rows")
    else:
        print(f"✓ No duplicate rows to drop")

    # Summary
    print(f"\n{'BEFORE CLEANING':20} {initial_shape[0]:>10,} rows × {initial_shape[1]:>3} columns")
    print(f"{'AFTER CLEANING':20} {combined_df.shape[0]:>10,} rows × {combined_df.shape[1]:>3} columns")
    print(f"{'ROWS REMOVED':20} {initial_shape[0] - combined_df.shape[0]:>10,}")
    print(f"{'COLUMNS REMOVED':20} {initial_shape[1] - combined_df.shape[1]:>10}")


    # ## Save Cleaned Combined Data

    # In[4]:


    # Save to a new CSV file
    output_file = 'combined_data.csv'
    combined_df.to_csv(output_file, index=False)

    print("=" * 60)
    print(f"✓ Saved cleaned combined data to: {output_file}")
    print("=" * 60)
    print(f"  Total rows: {len(combined_df):,}")
    print(f"  Total columns: {len(combined_df.columns)}")
    print(f"  Columns: {list(combined_df.columns)}")
    print(f"  Missing values remaining: {combined_df.isnull().sum().sum():,}")

    # Display first few rows
    print("\nFirst 5 rows of cleaned data:")
    combined_df.head()


"""Quick POS CSV analysis script."""
import pandas as pd

df = pd.read_csv('data/Brigade_Bangalore_10_April_26.csv')
print(f'Total rows: {len(df)}')
print(f'Unique invoice_numbers (raw): {df["invoice_number"].nunique()}')
print(f'invoice_type values: {df["invoice_type"].unique()}')
print(f'return_id non-null: {df["return_id"].notna().sum()}')
print(f'Sample customer_names: {df["customer_name"].head(10).tolist()}')
print(f'Sample dates: {df["order_date"].unique()[:5]}')
print(f'Total values: min={df["total_amount"].min()}, max={df["total_amount"].max()}')

# Apply filters
sales = df[df['invoice_type'].str.lower() == 'sales']
sales = sales[sales['return_id'].isna()]
sales = sales[sales['total_amount'] > 0]
print(f'After SALES + no-return + positive filter: {len(sales)} rows, {sales["invoice_number"].nunique()} invoices')

grouped = sales.groupby('invoice_number')['total_amount'].sum().reset_index()
gwp_dropped = grouped[grouped['total_amount'] >= 5]
print(f'After >=5 floor: {len(gwp_dropped)} invoices')
print(f'Invoice value range: min={gwp_dropped["total_amount"].min()}, max={gwp_dropped["total_amount"].max()}')

# Show identities — Guest vs named
named = df[df['customer_name'].str.strip().str.lower() != 'guest']
print(f'Named customers (unique): {named["customer_name"].str.strip().nunique()}')
print(f'Guest rows: {(df["customer_name"].str.strip().str.lower() == "guest").sum()}')

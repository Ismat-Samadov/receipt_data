#!/usr/bin/env python3
"""
Business Analytics Dashboard - Receipt Data Analysis
Generates visualizations and insights for executive decision-making
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('data/items.csv')

# Data preparation
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
df['subtotal'] = pd.to_numeric(df['subtotal'], errors='coerce')
df['line_total'] = pd.to_numeric(df['line_total'], errors='coerce')
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
df['cashless_payment'] = pd.to_numeric(df['cashless_payment'], errors='coerce')
df['cash_payment'] = pd.to_numeric(df['cash_payment'], errors='coerce')
df['bonus_payment'] = pd.to_numeric(df['bonus_payment'], errors='coerce')

# Extract time features
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%B')
df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.hour
df['day_of_week'] = df['date'].dt.day_name()

# Create charts directory
charts_dir = Path('charts')
charts_dir.mkdir(exist_ok=True)

# Helper function for consistent chart styling
def save_chart(filename, title):
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(charts_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {filename}")

# =============================================================================
# CHART 1: Revenue by Store Chain
# =============================================================================
print("\n📊 Generating business analytics charts...\n")

store_revenue = df.groupby('store_name')['subtotal'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(store_revenue)), store_revenue.values, color='steelblue', edgecolor='black', linewidth=0.7)
plt.xlabel('Store Chain', fontsize=12, fontweight='bold')
plt.ylabel('Total Revenue (AZN)', fontsize=12, fontweight='bold')
plt.xticks(range(len(store_revenue)), store_revenue.index, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.0f} AZN',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

save_chart('01_revenue_by_store.png', 'Revenue Performance by Store Chain')

# =============================================================================
# CHART 2: Monthly Revenue Trend
# =============================================================================
monthly_revenue = df.groupby('month_name')['subtotal'].sum().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]).dropna()

plt.figure(figsize=(12, 6))
plt.plot(range(len(monthly_revenue)), monthly_revenue.values, marker='o', linewidth=2.5,
         markersize=8, color='darkgreen')
plt.fill_between(range(len(monthly_revenue)), monthly_revenue.values, alpha=0.3, color='lightgreen')
plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('Revenue (AZN)', fontsize=12, fontweight='bold')
plt.xticks(range(len(monthly_revenue)), monthly_revenue.index, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add value labels
for i, value in enumerate(monthly_revenue.values):
    plt.text(i, value, f'{value:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

save_chart('02_monthly_revenue_trend.png', 'Monthly Revenue Trend')

# =============================================================================
# CHART 3: Payment Method Distribution
# =============================================================================
payment_data = pd.DataFrame({
    'Payment Method': ['Cashless', 'Cash', 'Bonus'],
    'Amount': [
        df['cashless_payment'].sum(),
        df['cash_payment'].sum(),
        df['bonus_payment'].sum()
    ]
})
payment_data = payment_data[payment_data['Amount'] > 0].sort_values('Amount', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(payment_data['Payment Method'], payment_data['Amount'],
               color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black', linewidth=0.7)
plt.xlabel('Payment Method', fontsize=12, fontweight='bold')
plt.ylabel('Total Amount (AZN)', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Add percentage and value labels
total = payment_data['Amount'].sum()
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = (height / total) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.0f} AZN\n({percentage:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

save_chart('03_payment_method_distribution.png', 'Customer Payment Preferences')

# =============================================================================
# CHART 4: Top 15 Products by Revenue
# =============================================================================
product_revenue = df.groupby('item_name')['line_total'].sum().sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(product_revenue)), product_revenue.values, color='coral', edgecolor='black', linewidth=0.7)
plt.xlabel('Revenue (AZN)', fontsize=12, fontweight='bold')
plt.ylabel('Product', fontsize=12, fontweight='bold')
plt.yticks(range(len(product_revenue)), product_revenue.index, fontsize=9)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2.,
             f' {width:,.1f} AZN',
             ha='left', va='center', fontweight='bold', fontsize=9)

save_chart('04_top_products_by_revenue.png', 'Top 15 Revenue-Generating Products')

# =============================================================================
# CHART 5: Shopping Time Analysis (Peak Hours)
# =============================================================================
hourly_transactions = df.groupby('hour')['filename'].nunique().sort_index()

plt.figure(figsize=(12, 6))
bars = plt.bar(hourly_transactions.index, hourly_transactions.values,
               color='purple', edgecolor='black', linewidth=0.7, alpha=0.8)
plt.xlabel('Hour of Day', fontsize=12, fontweight='bold')
plt.ylabel('Number of Transactions', fontsize=12, fontweight='bold')
plt.xticks(range(0, 24))
plt.grid(axis='y', alpha=0.3)

# Highlight peak hours
peak_hours = hourly_transactions.nlargest(3).index
for i, bar in enumerate(bars):
    if bar.get_x() in peak_hours:
        bar.set_color('darkred')
        bar.set_alpha(1.0)

save_chart('05_peak_shopping_hours.png', 'Customer Traffic by Hour')

# =============================================================================
# CHART 6: Average Transaction Value by Day of Week
# =============================================================================
receipt_totals = df.groupby('filename')['subtotal'].first().reset_index()
receipt_dates = df.groupby('filename')['day_of_week'].first().reset_index()
day_analysis = receipt_totals.merge(receipt_dates, on='filename')

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_by_day = day_analysis.groupby('day_of_week')['subtotal'].mean().reindex(day_order).dropna()

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(avg_by_day)), avg_by_day.values,
               color='teal', edgecolor='black', linewidth=0.7)
plt.xlabel('Day of Week', fontsize=12, fontweight='bold')
plt.ylabel('Average Transaction Value (AZN)', fontsize=12, fontweight='bold')
plt.xticks(range(len(avg_by_day)), avg_by_day.index, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f} AZN',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

save_chart('06_avg_transaction_by_day.png', 'Average Transaction Value by Day of Week')

# =============================================================================
# CHART 7: Store Performance Comparison (Transaction Count vs Revenue)
# =============================================================================
store_metrics = df.groupby('store_name').agg({
    'filename': 'nunique',
    'subtotal': 'sum'
}).rename(columns={'filename': 'transactions', 'subtotal': 'revenue'}).sort_values('revenue', ascending=False).head(10)

fig, ax1 = plt.subplots(figsize=(12, 6))

x_pos = range(len(store_metrics))
ax1.bar(x_pos, store_metrics['revenue'], alpha=0.7, color='steelblue',
        edgecolor='black', linewidth=0.7, label='Revenue')
ax1.set_xlabel('Store Chain', fontsize=12, fontweight='bold')
ax1.set_ylabel('Revenue (AZN)', color='steelblue', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(store_metrics.index, rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.plot(x_pos, store_metrics['transactions'], color='darkred', marker='o',
         linewidth=2.5, markersize=8, label='Transactions')
ax2.set_ylabel('Number of Transactions', color='darkred', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='darkred')

fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88), frameon=True, shadow=True)
plt.grid(True, alpha=0.3)
save_chart('07_store_performance_comparison.png', 'Store Performance: Revenue vs Transaction Volume')

# =============================================================================
# CHART 8: Price Range Distribution
# =============================================================================
price_bins = [0, 1, 5, 10, 20, 50, 100, 500]
price_labels = ['0-1', '1-5', '5-10', '10-20', '20-50', '50-100', '100+']
df['price_range'] = pd.cut(df['unit_price'], bins=price_bins, labels=price_labels, include_lowest=True)
price_dist = df['price_range'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(price_dist)), price_dist.values,
               color='orange', edgecolor='black', linewidth=0.7)
plt.xlabel('Price Range (AZN)', fontsize=12, fontweight='bold')
plt.ylabel('Number of Products', fontsize=12, fontweight='bold')
plt.xticks(range(len(price_dist)), price_dist.index)
plt.grid(axis='y', alpha=0.3)

# Add percentage labels
total_products = price_dist.sum()
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = (height / total_products) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

save_chart('08_price_range_distribution.png', 'Product Price Distribution')

# =============================================================================
# CHART 9: Monthly Transaction Count
# =============================================================================
monthly_transactions = df.groupby('month_name')['filename'].nunique().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]).dropna()

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(monthly_transactions)), monthly_transactions.values,
               color='mediumseagreen', edgecolor='black', linewidth=0.7)
plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('Number of Transactions', fontsize=12, fontweight='bold')
plt.xticks(range(len(monthly_transactions)), monthly_transactions.index, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

save_chart('09_monthly_transaction_count.png', 'Customer Activity: Monthly Transaction Volume')

# =============================================================================
# CHART 10: Product Category Performance (Top Categories by Revenue)
# =============================================================================
# Simplified category extraction based on common product naming patterns
def categorize_product(name):
    if pd.isna(name):
        return 'Other'
    name = str(name).upper()

    if any(word in name for word in ['YOQURT', 'SUD', 'PENDIR', 'AKTIVIA', 'CHEESE']):
        return 'Dairy Products'
    elif any(word in name for word in ['BANAN', 'POMIDOR', 'NEKTARIN', 'LIMON', 'GOYCE']):
        return 'Fresh Produce'
    elif any(word in name for word in ['COREK', 'TENDIR', 'BREAD', 'XONCA']):
        return 'Bakery'
    elif any(word in name for word in ['KOFE', 'ICIM', 'SIRAB', 'SU', 'WATER']):
        return 'Beverages'
    elif any(word in name for word in ['CEREZ', 'DONDURMA', 'SHOKOLAD', 'MERCI']):
        return 'Snacks & Sweets'
    elif any(word in name for word in ['ARIEL', 'DOMESTOS', 'SAMPUN', 'SHAMPUN', 'SABUN']):
        return 'Household & Personal Care'
    elif any(word in name for word in ['PAKET', 'BAG']):
        return 'Bags & Packaging'
    else:
        return 'Other'

df['category'] = df['item_name'].apply(categorize_product)
category_revenue = df.groupby('category')['line_total'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
bars = plt.barh(range(len(category_revenue)), category_revenue.values,
                color='indianred', edgecolor='black', linewidth=0.7)
plt.xlabel('Revenue (AZN)', fontsize=12, fontweight='bold')
plt.ylabel('Product Category', fontsize=12, fontweight='bold')
plt.yticks(range(len(category_revenue)), category_revenue.index)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# Add value labels and percentages
total_revenue = category_revenue.sum()
for i, bar in enumerate(bars):
    width = bar.get_width()
    percentage = (width / total_revenue) * 100
    plt.text(width, bar.get_y() + bar.get_height()/2.,
             f' {width:,.0f} AZN ({percentage:.1f}%)',
             ha='left', va='center', fontweight='bold', fontsize=9)

save_chart('10_category_revenue_performance.png', 'Revenue by Product Category')

# =============================================================================
# Generate Summary Statistics
# =============================================================================
print("\n" + "="*60)
print("📈 BUSINESS METRICS SUMMARY")
print("="*60)

total_revenue = df['subtotal'].sum()
total_transactions = df['filename'].nunique()
avg_transaction = total_revenue / total_transactions
total_items_sold = df['quantity'].sum()

print(f"\n💰 Total Revenue: {total_revenue:,.2f} AZN")
print(f"🛒 Total Transactions: {total_transactions:,}")
print(f"📊 Average Transaction Value: {avg_transaction:.2f} AZN")
print(f"📦 Total Items Sold: {total_items_sold:,.0f}")
print(f"🏪 Unique Stores: {df['store_name'].nunique()}")
print(f"🛍️  Unique Products: {df['item_name'].nunique()}")

print(f"\n💳 Payment Breakdown:")
print(f"   Cashless: {df['cashless_payment'].sum():,.2f} AZN ({df['cashless_payment'].sum()/total_revenue*100:.1f}%)")
print(f"   Cash: {df['cash_payment'].sum():,.2f} AZN ({df['cash_payment'].sum()/total_revenue*100:.1f}%)")
print(f"   Bonus: {df['bonus_payment'].sum():,.2f} AZN ({df['bonus_payment'].sum()/total_revenue*100:.1f}%)")

peak_hour = hourly_transactions.idxmax()
print(f"\n⏰ Peak Shopping Hour: {peak_hour}:00")
print(f"📅 Most Active Month: {monthly_transactions.idxmax()}")

print("\n" + "="*60)
print("✅ All charts generated successfully in charts/ directory")
print("="*60 + "\n")

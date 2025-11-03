import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from db_service import GetData
import plotly.graph_objects as go

def transaction_dynamics():

    df = GetData("aggregated_transaction")

    # Convert counts and amounts to a more readable scale for visualization
    df['TransactionCount'] = df['TransactionCount'] / 1000000  # Convert to millions
    df['TransactionAmount'] = df['TransactionAmount'] / 1000000000  # Convert to billions

    # --- 2. USER'S VISUALIZATION LOGIC ---

    df['Year_Quarter'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)

    # Set a consistent style for better visuals
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    st.markdown("---")

    # Chart 1: Time Series Analysis
    st.header("1. Total Transaction Count Over Time (Quarterly)")
    time_series_data = df.groupby('Year_Quarter')['TransactionCount'].sum().reset_index()

    time_series_data['sort_key'] = time_series_data['Year_Quarter'].apply(
        lambda x: int(x.split('-')[0]) * 4 + int(x.split('-Q')[1])
    )
    time_series_data = time_series_data.sort_values(by='sort_key')

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=time_series_data, x='Year_Quarter', y='TransactionCount', marker='o', color='skyblue', ax=ax1)
    ax1.set_title('1. Total Transaction Count Over Time (Quarterly)', fontsize=14)
    ax1.set_xlabel('Year-Quarter')
    ax1.set_ylabel('Total Transaction Count (in Millions)')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    # Chart 2: Top 5 States by Volume
    st.header("2. Top 5 States by Total Transaction Count")
    state_volume = df.groupby('State')['TransactionCount'].sum().nlargest(5).reset_index()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=state_volume, x='State', y='TransactionCount', palette='viridis', ax=ax2)
    ax2.set_title('2. Top 5 States by Total Transaction Count', fontsize=14)
    ax2.set_xlabel('State')
    ax2.set_ylabel('Total Transaction Count (in Millions)')
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # Chart 3: Distribution of Transaction Amount by Type
    st.header("3. Distribution of Total Transaction Amount by Transaction Type")
    type_distribution = df.groupby('TransactionType')['TransactionAmount'].sum().reset_index()

    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ax3.pie(
        type_distribution['TransactionAmount'],
        labels=type_distribution['TransactionType'],
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette('pastel')
    )
    ax3.set_title('3. Distribution of Total Transaction Amount by Transaction Type', fontsize=14)
    ax3.axis('equal')
    st.pyplot(fig3)

    # Chart 4: Average Transaction Value (ATV) for Top State
    st.header("4. Average Transaction Value (ATV) Quarterly in top state")
    top_state = df.groupby('State')['TransactionCount'].sum().idxmax()
    st.info(f"Analysis Focus: Top Volume State is **{top_state}**")

    top_state_df = df[df['State'] == top_state].copy()

    quarterly_atv = top_state_df.groupby('Year_Quarter').agg(
        TotalAmount=('TransactionAmount', 'sum'),
        TotalCount=('TransactionCount', 'sum')
    ).reset_index()

    # Handle division by zero for ATV calculation (in case Count is zero, which is unlikely with )
    quarterly_atv['ATV'] = quarterly_atv.apply(
        lambda row: row['TotalAmount'] / row['TotalCount'] if row['TotalCount'] > 0 else 0,
        axis=1
    )

    quarterly_atv['sort_key'] = quarterly_atv['Year_Quarter'].apply(
        lambda x: int(x.split('-')[0]) * 4 + int(x.split('-Q')[1])
    )
    quarterly_atv = quarterly_atv.sort_values(by='sort_key')

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=quarterly_atv, x='Year_Quarter', y='ATV', marker='o', color='indianred', ax=ax4)
    ax4.set_title(f'4. Average Transaction Value (ATV) Quarterly in {top_state}', fontsize=14)
    ax4.set_xlabel('Year-Quarter')
    ax4.set_ylabel('Average Transaction Value (Amount / Count)')
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

    # Chart 5: Top 10 States Volume (Count) vs. Value (Amount) - Latest Quarter
    st.header("5. State Metrics: Volume (Count) vs. Value (Amount) in latest year and latest quarter")
    latest_year = df['Year'].max()
    latest_quarter = df[df['Year'] == latest_year]['Quarter'].max()

    latest_df = df[(df['Year'] == latest_year) & (df['Quarter'] == latest_quarter)]

    state_metrics = latest_df.groupby('State').agg(
        TotalCount=('TransactionCount', 'sum'),
        TotalAmount=('TransactionAmount', 'sum')
    ).reset_index()


    top_10_states = state_metrics.nlargest(10, 'TotalCount')

    fig5, ax1_5 = plt.subplots(figsize=(14, 7))

    sns.barplot(
        x='State', y='TotalCount', data=top_10_states,
        ax=ax1_5, color='lightgreen', alpha=0.6, label='Total Count'
    )
    ax1_5.set_ylabel('Total Transaction Count (Millions)', color='green')
    ax1_5.tick_params(axis='y', labelcolor='green')
    ax1_5.set_xlabel('State')
    ax1_5.set_xticklabels(ax1_5.get_xticklabels(), rotation=45, ha='right')

    ax2_5 = ax1_5.twinx()
    sns.lineplot(
        x='State', y='TotalAmount', data=top_10_states,
        ax=ax2_5, marker='o', color='purple', label='Total Amount (Billions)'
    )
    ax2_5.set_ylabel('Total Transaction Amount (Billions)', color='purple')
    ax2_5.tick_params(axis='y', labelcolor='purple')

    ax1_5.set_title(f'5. State Metrics: Volume (Count) vs. Value (Amount) in {latest_year}-Q{latest_quarter}',
                    fontsize=14)
    st.pyplot(fig5)

    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(df)





def device_dominance():
    st.title("2) 📱 Device Dominance and User Engagement Analysis")
    st.markdown("---")
    st.markdown("""
        Examine user engagement metrics and the distribution of transactions by device type/brand.
        **Key Visuals:**
        * Quarterly growth of new vs. existing users.
        * Market share of top mobile brands used for transactions.
    """)
    df = GetData("aggregated_user")
    aggregated_user = df

    aggregated_user['Year'] = aggregated_user['Year'].astype(int)

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    st.header("1. Top 5 Mobile Brands by Total Registered User Count")

    # 1. Which 5 mobile brands have the highest total number of registered users (UserCount)?
    top_5_brands = aggregated_user.groupby('MobileBrand')['UserCount'].sum().nlargest(5).sort_values(ascending=False)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_5_brands.index, y=top_5_brands.values, palette='viridis', ax=ax1)
    ax1.set_title('Top 5 Mobile Brands by Total Registered User Count')
    ax1.set_xlabel('Mobile Brand')
    ax1.set_ylabel('Total User Count (in millions)')
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x / 1000000)) + "M")
    )
    st.pyplot(fig1)

    st.header("2. Distribution of User Engagement Percentage")

    # 2. What is the distribution of the Percentage column across all devices?
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.histplot(aggregated_user['Percentage'], bins=10, kde=True, color='skyblue', ax=ax2)
    ax2.set_title('Distribution of User Engagement Percentage')
    ax2.set_xlabel('Percentage (%)')
    ax2.set_ylabel('Frequency (Count of Records)')
    st.pyplot(fig2)

    st.header("3. Total Registered User Count Over the Years")

    # 3. How has the total UserCount changed over the Year?
    user_count_over_time = aggregated_user.groupby('Year')['UserCount'].sum()

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    user_count_over_time.plot(kind='line', marker='o', color='crimson', ax=ax3)
    ax3.set_title('Total Registered User Count Over the Years')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Total User Count (in millions)')
    ax3.set_xticks(user_count_over_time.index)
    ax3.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x / 1000000)) + "M")
    )
    ax3.grid(axis='y', linestyle='--')
    st.pyplot(fig3)

    st.header("4. Average Engagement Percentage for Top 3 Mobile Brands")

    # 4. How does the average Percentage differ between the top 3 mobile brands with the highest UserCount?
    top_3_brands = aggregated_user.groupby('MobileBrand')['UserCount'].sum().nlargest(3).index.tolist()

    top_brands_df = aggregated_user[aggregated_user['MobileBrand'].isin(top_3_brands)]

    avg_percentage = top_brands_df.groupby('MobileBrand')['Percentage'].mean().sort_values(ascending=False)

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=avg_percentage.index, y=avg_percentage.values, palette='magma', ax=ax4)
    ax4.set_title('Average Engagement Percentage for Top 3 Mobile Brands')
    ax4.set_xlabel('Mobile Brand')
    ax4.set_ylabel('Average Percentage (%)')
    # for index, value in enumerate(avg_percentage.values):
    #     ax4.text(index, value + 0.5, f'{value:.2f}%', ha='center')
    ax4.set_ylim(0, avg_percentage.max() * 1.2)
    st.pyplot(fig4)

    st.header("5. Relationship between User Count and Engagement Percentage")

    # 5. Is there a correlation between UserCount and Percentage?
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='UserCount', y='Percentage', data=aggregated_user, hue='MobileBrand', size='UserCount',
                    sizes=(20, 200), alpha=0.7, palette='tab10', ax=ax5)
    ax5.set_title('Relationship between User Count and Engagement Percentage')
    ax5.set_xlabel('Registered User Count')
    ax5.set_ylabel('Percentage (%)')

    ax5.ticklabel_format(style='plain', axis='x')
    ax5.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )

    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig5)

    correlation = aggregated_user['UserCount'].corr(aggregated_user['Percentage'])
    st.code(f"Pearson Correlation between UserCount and Percentage: {correlation:.2f}")

    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(aggregated_user)


def insurance_analysis():
    st.title("3) 🛡️ Insurance Penetration and Growth Potential Analysis")
    st.markdown("---")
    st.markdown("""
        Investigate the adoption rate of insurance products across different regions.
        **Key Visuals:**
        * State-wise heat map of insurance transaction amount.
        * Growth of insurance product categories (e.g., term, health, motor).
    """)
    st.warning(
        "Note: This analysis typically requires filtering a 'Transaction Type' column for 'Insurance' or a separate 'Aggregated Insurance' table.")

    df = GetData("aggregated_insurance")
    #
    # # Calculate combined time period
    # df['Time_Period'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)
    #
    # # Convert amounts to billions for plotting readability
    # df['TransactionAmount'] = df['TransactionAmount'] / 1000000000

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Quarter'] = pd.to_numeric(df['Quarter'], errors='coerce')
    df['TransactionCount'] = pd.to_numeric(df['TransactionCount'], errors='coerce')
    df['TransactionAmount'] = pd.to_numeric(df['TransactionAmount'], errors='coerce')

    # Drop rows with missing values that are critical for analysis
    df.dropna(inplace=True)

    # Create a combined time axis for better time-series plotting (e.g., '2019-Q1')
    df['Time_Period'] = df['Year'].astype(int).astype(str) + '-Q' + df['Quarter'].astype(int).astype(str)
    # To ensure correct time sorting, sort the dataframe once
    df.sort_values(by=['Year', 'Quarter'], inplace=True)

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    st.markdown("---")
    st.header("1. Overall Insurance Transaction Amount Trend Over Time")

    # ==============================================================================
    # --- Q1: Overall Growth Trend (Line Plot) ---
    # ==============================================================================

    q1_data = df.groupby('Time_Period')['TransactionAmount'].sum().reset_index()

    # Create a sort key for chronological plotting
    q1_data['sort_key'] = q1_data['Time_Period'].apply(
        lambda x: int(x.split('-')[0]) * 4 + int(x.split('-Q')[1])
    )
    q1_data = q1_data.sort_values(by='sort_key')

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(q1_data['Time_Period'], q1_data['TransactionAmount'], marker='o', linestyle='-', color='tab:blue')
    ax1.set_title('1. Overall Insurance Transaction Amount Trend Over Time', fontsize=14)
    ax1.set_xlabel('Time Period (Year-Quarter)', fontsize=12)
    ax1.set_ylabel('Total Transaction Amount (in Billions)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)# ha='right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig1)

    st.header(f"2. State-wise Transaction Amount in Latest Quarter")

    # ==============================================================================
    # --- Q2: State-wise Penetration (Latest Quarter - Bar Chart) ---
    # ==============================================================================

    latest_year = df['Year'].max()
    latest_quarter = df[df['Year'] == latest_year]['Quarter'].max()

    q2_latest_data = df[(df['Year'] == latest_year) & (df['Quarter'] == latest_quarter)]
    q2_data = q2_latest_data.groupby('State')['TransactionAmount'].sum().sort_values(ascending=False).reset_index()

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.bar(q2_data['State'], q2_data['TransactionAmount'], color='tab:green')
    ax2.set_title(f'2. State-wise Transaction Amount in Latest Quarter ({latest_year}-Q{latest_quarter})', fontsize=14)
    ax2.set_xlabel('State', fontsize=12)
    ax2.set_ylabel('Total Transaction Amount (in Billions)', fontsize=12)
    ax2.tick_params(axis='x', rotation=90)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig2)

    # --- 4. Quarterly Count Distribution (Bar Plot) ---
    st.subheader("3. Total Transaction Count Distribution by Quarter")
    quarterly_counts = df.groupby('Quarter')['TransactionCount'].sum().reset_index()

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=quarterly_counts, x='Quarter', y='TransactionCount', palette='Paired', ax=ax4)

    ax4.set_title('Total Transaction Count Distribution by Quarter (All Years)', fontsize=14)
    ax4.set_xlabel('Quarter', fontsize=12)
    ax4.set_ylabel('Total Transaction Count', fontsize=12)
    ax4.set_xticks(quarterly_counts['Quarter'])
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    # ==============================================================================
    # --- Q4: State-wise Growth Rate (YoY - Bar Chart) ---
    # ==============================================================================
    st.header("4. State-wise Transaction Amount Growth Rate")
    first_year = df['Year'].min()
    last_year = df['Year'].max()

    q4_first_year_data = df[df['Year'] == first_year].groupby('State')['TransactionAmount'].sum().rename(
        'Amount_First_Year')
    q4_last_year_data = df[df['Year'] == last_year].groupby('State')['TransactionAmount'].sum().rename(
        'Amount_Last_Year')

    q4_data = pd.merge(q4_first_year_data, q4_last_year_data, on='State', how='inner')

    q4_data['Growth_Rate'] = np.where(
        q4_data['Amount_First_Year'] != 0,
        ((q4_data['Amount_Last_Year'] - q4_data['Amount_First_Year']) / q4_data['Amount_First_Year']) * 100,
        0
    )

    q4_data.sort_values(by='Growth_Rate', ascending=False, inplace=True)

    fig4, ax4 = plt.subplots(figsize=(14, 7))
    colors = ['tab:orange' if x > 0 else 'tab:grey' for x in q4_data['Growth_Rate']]
    ax4.bar(q4_data.index, q4_data['Growth_Rate'], color=colors)
    ax4.set_title(f'4. State-wise Transaction Amount Growth Rate ({first_year} to {last_year})', fontsize=14)
    ax4.set_xlabel('State', fontsize=12)
    ax4.set_ylabel('Percentage Growth Rate (%)', fontsize=12)
    ax4.tick_params(axis='x', rotation=90)
    ax4.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig4)

    st.header("5. Quarterly Transaction Amount Trend for Top 5 States")

    # ==============================================================================
    # --- Q5: Quarterly Trend by Top States (Multi-Line Chart) ---
    # ==============================================================================

    top_5_states = df.groupby('State')['TransactionAmount'].sum().nlargest(5).index.tolist()

    q5_data = df[df['State'].isin(top_5_states)]

    q5_plot_data = q5_data.groupby(['State', 'Time_Period'])['TransactionAmount'].sum().reset_index()

    # Re-sort combined data for plotting
    q5_plot_data['sort_key'] = q5_plot_data['Time_Period'].apply(
        lambda x: int(x.split('-')[0]) * 4 + int(x.split('-Q')[1])
    )
    q5_plot_data = q5_plot_data.sort_values(by='sort_key')

    fig5, ax5 = plt.subplots(figsize=(12, 7))

    for state in top_5_states:
        state_data = q5_plot_data[q5_plot_data['State'] == state]
        ax5.plot(state_data['Time_Period'], state_data['TransactionAmount'], marker='o', label=state)

    ax5.set_title('5. Quarterly Transaction Amount Trend for Top 5 States', fontsize=14)
    ax5.set_xlabel('Time Period (Year-Quarter)', fontsize=12)
    ax5.set_ylabel('Total Transaction Amount (in Billions)', fontsize=12)
    ax5.tick_params(axis='x', rotation=45)
    ax5.legend(title='State')
    ax5.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig5)

    st.markdown("---")
    st.subheader("Raw Data Preview ")
    st.dataframe(df)
# --- 2. MAIN PAGE FUNCTIONS ---
def transaction_analysis():
    """Visualization for Transaction Analysis Across States and Districts."""
    st.title("4) 🗺️ Geo-Transaction Analysis: States & Districts")
    st.markdown("---")
    st.markdown("""
        Detailed analysis of transaction volume and value metrics broken down by geographical entities (states, districts, and pincodes).
        **Key Visuals:**
        * Volume vs. Value comparison for top districts.
        * Quarterly distribution of transaction amount.
    """)

    df = GetData("top_map")

    df['Year'] = df['Year'].astype(int)

    df.fillna(0, inplace=True)

    df = df.rename(columns={
        'Quater': 'Quarter',
        'TransacionCount': 'TransactionCount',
        'TransacionAmount': 'TransactionAmount'
    })
    # 2. Fill NULL EntityName with a placeholder for the states row where data is NULL
    df['EntityName'] = df['EntityName'].fillna(df['State'])
    # 3. Ensure transaction columns are numeric (coercing NaNs from the 'states' row)
    df['TransactionCount'] = pd.to_numeric(df['TransactionCount'], errors='coerce')
    df['TransactionAmount'] = pd.to_numeric(df['TransactionAmount'], errors='coerce')

    # --- Determine latest year (for Q

    # Define necessary variables for the user's code blocks
    latest_year = df['Year'].max()
    df_valid = df[df['EntityType'].isin(['districts', 'pincodes'])].copy()  # Filter to be explicit

    # Calculate state_total_amount (total amount across the valid dataset) for Q4/Q9
    state_total_amount = df_valid['TransactionAmount'].sum()

    st.info(f"All geographic visualizations focus on data spanning 2020 to {latest_year}.")
    st.markdown("---")

    st.header("1. Volume vs. Value Comparison for Top 5 Districts")

    # =================================================================
    # Q1: How do the Transaction Count and Amount compare for the top 5 districts in the latest year?
    # =================================================================
    df_districts_latest = df_valid[(df_valid['Year'] == latest_year) & (df_valid['EntityType'] == 'districts')]
    df_q1 = df_districts_latest.groupby('EntityName')[['TransactionCount', 'TransactionAmount']].sum().reset_index()

    # Sort by Amount (as a proxy for 'top')
    df_q1 = df_q1.sort_values(by='TransactionAmount', ascending=False).head(5)
    df_q1_melt = df_q1.melt(id_vars='EntityName', value_vars=['TransactionCount', 'TransactionAmount'],
                            var_name='Metric', value_name='Value')

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.barplot(x='EntityName', y='Value', hue='Metric', data=df_q1_melt, palette='Paired', ax=ax1)
    ax1.set_title(f'1. Volume vs. Value Comparison for Top 5 Districts ({latest_year})', fontsize=14)
    ax1.set_xlabel('District Name')
    ax1.set_ylabel('Value (Log Scale)')
    ax1.set_yscale('log')  # Log scale is essential for comparing counts (low) and amounts (high)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Metric')
    st.pyplot(fig1)

    st.header(f"2. Quarterly Distribution of Total Transaction Amount ({latest_year})")

    # =================================================================
    # Q2: How is the total Transaction Amount distributed across the four quarters in the latest year?
    # =================================================================
    df_q2 = df_valid[df_valid['Year'] == latest_year].groupby('Quarter')['TransactionAmount'].sum().reset_index()

    # Calculate percentages for the pie chart
    total_amount_latest_year = df_q2['TransactionAmount'].sum()
    df_q2['Percentage'] = (df_q2['TransactionAmount'] / total_amount_latest_year) * 100
    df_q2['QuarterLabel'] = 'Q' + df_q2['Quarter'].astype(str)

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.pie(df_q2['Percentage'], labels=df_q2['QuarterLabel'], autopct='%1.1f%%', startangle=90,
            colors=sns.color_palette('pastel'))
    ax2.set_title(f'2. Quarterly Distribution of Total Transaction Amount ({latest_year})', fontsize=14)
    st.pyplot(fig2)

    st.header("3. Total Volume vs. Total Value Comparison (Across All States)")

    # =================================================================
    # Q3: Volume vs. Value Leader Comparison (Top State)
    # =================================================================
    df_q3 = df_valid.groupby('State').agg(
        TotalCount=('TransactionCount', 'sum'),
        TotalAmount=('TransactionAmount', 'sum')
    ).reset_index()

    df_q3_melt = df_q3.melt(id_vars='State', value_vars=['TotalCount', 'TotalAmount'], var_name='Metric',
                            value_name='Value')

    fig3, ax3 = plt.subplots(figsize=(15, 6))
    sns.barplot(x='Metric', y='Value', hue='State', data=df_q3_melt, palette='tab10', ax=ax3)
    ax3.set_title('3. Total Volume vs. Total Value Comparison (All States)', fontsize=14)
    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Value (Log Scale)')
    ax3.set_yscale('log')  # Use log scale due to large magnitude difference
    ax3.legend(title='State')
    st.pyplot(fig3)

    st.header("4. Average Transaction Value (ATV) Dispersion (State vs. Top Pincodes)")

    # =================================================================
    # Q4: ATV Dispersion (State vs Top 3 Pincodes)
    # =================================================================
    # 1. State ATV (Overall ATV for the valid dataset)
    state_atv = state_total_amount / df_valid['TransactionCount'].sum()

    # 2. Top 3 Pincodes ATV
    df_pincode_atv = df_valid[df_valid['EntityType'] == 'pincodes'].groupby('EntityName').agg(
        TotalCount=('TransactionCount', 'sum'),
        TotalAmount=('TransactionAmount', 'sum')
    )
    df_pincode_atv['ATV'] = df_pincode_atv['TotalAmount'] / df_pincode_atv['TotalCount']
    df_pincode_atv = df_pincode_atv.sort_values(by='TotalCount', ascending=False).head(3).reset_index()

    # 3. Combine Data
    df_q4_plot = pd.DataFrame({
        'Entity': ['Overall State ATV'] + list(df_pincode_atv['EntityName']),
        'ATV': [state_atv] + list(df_pincode_atv['ATV'])
    })

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Entity', y='ATV', data=df_q4_plot, palette='Spectral', ax=ax4)
    ax4.set_title('4. Average Transaction Value (ATV) Dispersion', fontsize=14)
    ax4.set_xlabel('Entity')
    ax4.set_ylabel('Average Transaction Value (Billion/Count)')
    for index, row in df_q4_plot.iterrows():
        ax4.text(index, row['ATV'], f'{row["ATV"]:.3f}', ha='center', va='bottom')
    st.pyplot(fig4)

    st.header(f"5. Quarterly Growth in Transaction Amount (Q2 vs Q3, {latest_year})")

    # =================================================================
    # Q5: Which state showed the largest Year-over-Year (YoY) growth percentage in Transaction Amount from 2020 to 2022?
    # Note: Adapted to show QoQ growth (Q3 vs Q2)
    # =================================================================
    df_states_agg = df_valid[df_valid['Year'] == latest_year].groupby(['State', 'Quarter'])[
        'TransactionAmount'].sum().reset_index()

    # Pivot for comparison
    df_pivot = df_states_agg.pivot_table(index='State', columns='Quarter', values='TransactionAmount', aggfunc='sum')
    q2_val = 2
    q3_val = 3

    # Calculate QoQ Growth (Q3 vs Q2)
    if q2_val in df_pivot.columns and q3_val in df_pivot.columns:
        df_pivot['QoQ_Growth'] = ((df_pivot[q3_val] - df_pivot[q2_val]) / df_pivot[q2_val]) * 100
    else:
        # Handle case where quarters don't exist in the
        df_pivot['QoQ_Growth'] = 0
        st.warning(f"Could not calculate QoQ growth for Q{q2_val} to Q{q3_val}. Showing 0% growth as fallback.")

    df_q5 = df_pivot.reset_index().sort_values(by='QoQ_Growth', ascending=False)

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='QoQ_Growth', y='State', data=df_q5, palette='coolwarm', ax=ax5)
    ax5.set_title(f'5. Quarterly Growth in Transaction Amount (Q{q2_val} to Q{q3_val}, {latest_year})', fontsize=14)
    ax5.set_xlabel('Quarterly Growth (%)')
    ax5.set_ylabel('State')
    for index, row in df_q5.iterrows():
        ax5.text(row['QoQ_Growth'], index, f'{row["QoQ_Growth"]:.1f}%', va='center')
    st.pyplot(fig5)

    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(df)

def home_page():
    """The landing page of the application, featuring a Plotly Choropleth map with dynamic filters."""
    # --- Configuration Mapping (Moved inside the function to ensure definition) ---
    METRIC_OPTIONS = {
        "Total Registered Users": {
            "table": "map_user",
            "column": "RegisteredUsers",
            "metric_label": "Total Registered Users",
            "data_label": "Registered Users"
        },
        "Total Transaction Amount (Map)": {
            "table": "map_map",
            "column": "MetricAmount",
            "metric_label": "Total Transaction Amount (Map)",
            "data_label": "Transaction Amount"
        },
        "Total Insurance Amount": {
            "table": "map_insurance",
            "column": "MetricAmount",
            "metric_label": "Total Insurance Amount",
            "data_label": "Insurance Amount"
        }
    }

    # The URL for the GeoJSON file (used in the original code)
    INDIA_STATES_GEOJSON_URL = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"

    st.set_page_config(layout="wide")
    st.title("Welcome to the PhonePe Data Visualization App 📊")
    st.markdown("---")
    st.header("Overview")
    st.info("Use the sidebar on the left to navigate and select the data metric.")
    st.subheader("Goal:")
    st.write(
        "This application provides interactive visualizations to decode critical business patterns by allowing dynamic selection of user, transaction, and insurance metrics.")

    # --- Sidebar Filters ---
    st.sidebar.header("Data & Map Filters")

    # 1. Data Category Selector (NEW)
    selected_metric_key = st.sidebar.radio(
        "Select Data Metric:",
        options=list(METRIC_OPTIONS.keys()),
        index=0  # Default to Registered Users
    )

    # Get dynamic configuration
    config = METRIC_OPTIONS[selected_metric_key]
    table_name = config['table']
    metric_column = config['column']
    metric_label = config['metric_label']
    data_label = config['data_label']

    # --- Data Loading and Cleaning ---
    df = GetData(table_name)  # Dynamic table name load

    # Ensure Year is numeric for filtering
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

    # Standardize State names for GeoJSON matching
    df['State'] = df['State'].str.lower().str.replace('-', ' ').str.replace('&', 'and').str.title()
    # Correcting the Andaman & Nicobar Island state name to match GeoJSON (this is the key step)
    df['State'] = df['State'].replace('Andaman And Nicobar Islands', 'Andaman & Nicobar Island')
    df['State'] = df['State'].replace('Dadra And Nagar Haveli And Daman And Diu',
                                      'Dadra and Nagar Haveli and Daman and Diu')

    # 2. Year Selector (Multiselect)
    all_years = sorted(df['Year'].unique().tolist(), reverse=True)
    selected_years = st.sidebar.multiselect(
        "Select Year(s):",
        options=all_years,
        default=[all_years[0]] if all_years else []  # Default to the latest year
    )

    # 3. Quarter Selector (Multiselect)
    all_quarters = sorted(df['Quater'].unique().tolist())
    selected_quarters = st.sidebar.multiselect(
        "Select Quarter(s):",
        options=all_quarters,
        default=all_quarters  # Default to all quarters
    )

    # 4. State Selector (Selectbox)
    all_states = ['All States'] + sorted(df['State'].unique().tolist())
    selected_state = st.sidebar.selectbox(
        "Select State:",
        options=all_states,
        index=0
    )

    # --- Apply Final Filters ---
    df_filtered = df.copy()

    # Filter by Year
    if selected_years:
        df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]

    # Filter by Quarter
    if selected_quarters:
        df_filtered = df_filtered[df_filtered['Quater'].isin(selected_quarters)]

    # Filter by State
    if selected_state != 'All States':
        df_filtered = df_filtered[df_filtered['State'] == selected_state]

    # --- Aggregate Data for Map ---
    # Aggregate by 'State' using the dynamic metric_column
    if metric_column not in df_filtered.columns:
        st.error(
            f"Error: The selected metric column '{metric_column}' is not available in the data from '{table_name}'.")
        return

    df_map_data = df_filtered.groupby('State')[metric_column].sum().reset_index()

    if df_map_data.empty:
        st.warning("No data available for the selected filters.")
        return

    # Rename the aggregated column to a consistent name for plotting
    PLOT_COLUMN = 'PlotValue'
    df_map_data = df_map_data.rename(columns={metric_column: PLOT_COLUMN})

    # --- Map Visualization Logic ---

    # Dynamic Title
    title_parts = [metric_label, " by State"]

    if selected_state != 'All States':
        title_parts.append(f" ({selected_state})")

    quarter_str = ', '.join(map(str, selected_quarters))
    year_str = ', '.join(map(str, selected_years))

    title_parts.append(f" | Quarters: {quarter_str} | Years: {year_str}")
    map_title = "".join(title_parts)

    # Custom hover template and colorbar title based on metric type
    if 'Amount' in metric_label:
        hovertemplate_text = '<b>%{location}</b><br>Total: ₹%{z:,.2f}<extra></extra>'
        colorbar_title = {'text': f"{data_label} (₹)"}
    else:  # Count/Users
        hovertemplate_text = '<b>%{location}</b><br>Total: %{z:,}<extra></extra>'
        colorbar_title = {'text': data_label}

    fig = go.Figure(data=go.Choropleth(
        geojson=INDIA_STATES_GEOJSON_URL,
        featureidkey='properties.ST_NM',
        locationmode='geojson-id',
        locations=df_map_data['State'],
        z=df_map_data[PLOT_COLUMN],  # Use the consistent plot column
        zauto=True,
        colorscale='Viridis',
        marker_line_color='black',
        hovertemplate=hovertemplate_text,
        colorbar=dict(
            title=colorbar_title,
            thickness=15,
            len=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            xanchor='left',
            x=0.01,
            yanchor='bottom',
            y=0.1
        )
    ))

    # Update map configuration
    fig.update_geos(
        visible=False,
        fitbounds="locations",
        projection=dict(
            type='conic conformal',
            parallels=[12, 35],
            rotation={'lat': 24, 'lon': 80}
        ),
        lonaxis={'range': [68, 98]},
        lataxis={'range': [6, 38]}
    )

    # Update layout configuration
    fig.update_layout(
        title=dict(
            text=map_title,
            xanchor='center',
            x=0.5,
            yref='paper',
            yanchor='bottom',
            y=1,
            pad={'b': 10}
        ),
        margin={'r': 0, 't': 30, 'l': 0, 'b': 0},
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Display the final aggregated data
    st.subheader(f"Data Summary for Map ({metric_label})")
    st.write(
        f"The map displays the aggregated **{metric_label}** for the selected filters, using data from the **{table_name}** table and aggregating the **{metric_column}** column."
    )
    st.dataframe(df_map_data.rename(columns={PLOT_COLUMN: metric_label}))


def data_visualization_page():
    # import pdb
    # pdb.set_trace()
    """
    Handles the navigation for Aggregated, Map, and Top data pages.
    It includes the sub-navigation for the first option.
    """
    # Use st.session_state to handle navigation if needed, but for simplicity,
    # we'll use the main sidebar to differentiate the pages.

    # ----------------------------------------------------
    # CONDITIONALLY DISPLAY SUB-NAVIGATION
    # This logic assumes the main sidebar selection (e.g., "Aggregated Data Visualization")
    # dictates what content to show.
    # ----------------------------------------------------

    st.sidebar.markdown("### Visualization Focus:")

    # Check the main selection in the sidebar to determine which sub-menu to show
    selection = st.session_state.get('main_selection', 'Home')

    if selection == "Aggregated Data Visualization":
        st.title("Aggregated Data Visualization")

        # --- SUB-NAVIGATION RADIO FOR AGGREGATED DATA ---
        agg_options = {
            "Decoding Transaction Dynamics on PhonePe": transaction_dynamics,
            "Device Dominance and User Engagement Analysis": device_dominance,
            "Insurance Penetration and Growth Potential Analysis": insurance_analysis,
        }

        sub_selection = st.sidebar.radio(
            "Aggregated Reports:",
            list(agg_options.keys()),
            key='agg_report_selection1'  # Unique key for this widget
        )

        # Call the selected sub-page function
        agg_options[sub_selection]()

    elif selection == "Map Data Visualization":
        st.title("Map Data Visualization 🗺️")

        agg_options = {
            "Transaction Analysis for Market Expansion": analyze_market_transactions,
            "User Engagement and Growth Strategy": analyze_user_engagement,
            "Insurance Engagement Analysis": analyze_insurance_engagement,
        }

        sub_selection = st.sidebar.radio(
            "Aggregated Reports:",
            list(agg_options.keys()),
            key='agg_report_selection2'  # Unique key for this widget
        )

        agg_options[sub_selection]()

    elif selection == "Top Data Visualization":
        st.title("Top Data Visualization 🥇")
        # --- SUB-NAVIGATION RADIO FOR AGGREGATED DATA ---
        agg_options = {
            "Transaction Analysis Across States and Districts": transaction_analysis,
            "User Registration Analysis": analyze_user_registration,
            "Insurance Transactions Analysis": insurance_Transactions_analysis,
        }

        sub_selection = st.sidebar.radio(
            "Top Data Visualization Reports:",
            list(agg_options.keys()),
            key='agg_report_selection3'  # Unique key for this widget
        )

        agg_options[sub_selection]()


def insurance_Transactions_analysis():
    """Detailed Insurance Engagement Analysis including Top States, Quarterly Trends, and Breakdown."""
    st.title("5) 🏦 Insurance Engagement Analysis")
    st.markdown("---")
    st.markdown("""
        A focused analysis on insurance engagement metrics:
        * Identifying the top 10 contributing states by transaction count and amount.
        * Tracking the quarterly growth trend.
        * Breaking down transaction count by insurance type for the top states.
    """)
    df = GetData("top_insurance")



    st.markdown("---")

    # =================================================================
    # VIZ 1: Top 10 States by Total Transaction Count
    # =================================================================
    st.header("1. Top 10 States by Total Insurance Transaction Count")
    # import pdb
    # pdb.set_trace()
    state_transaction_count = df.groupby('State')['TransactionCount'].sum()
    top_10_states = state_transaction_count.sort_values(ascending=False).head(10)

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    plt.style.use('ggplot')  # Apply style temporarily

    # Create the horizontal bar chart
    top_10_states.sort_values(ascending=True).plot(kind='barh', color='#0077b6', ax=ax1)

    # Add labels and title
    ax1.set_title('Top 10 States by Total Insurance Transaction Count', fontsize=16, pad=20)
    ax1.set_xlabel('Total Transaction Count (in Millions)', fontsize=12)
    ax1.set_ylabel('State', fontsize=12)

    # Add data labels to the bars
    for index, value in enumerate(top_10_states.sort_values(ascending=True).values):
        ax1.text(value, index, f'{value:,.2f}M', va='center', ha='left')

    ax1.tick_params(axis='x', rotation=0)
    st.pyplot(fig1)

    st.markdown("---")

    # =================================================================
    # VIZ 2: Top 10 States by Total Transaction Amount
    # =================================================================
    st.header("2. Top 10 States by Total Insurance Transaction Amount")

    state_transaction_amount = df.groupby('State')['TransactionAmount'].sum()
    top_10_states_amount = state_transaction_amount.sort_values(ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')  # Apply style temporarily

    # Create the horizontal bar chart, sorted ascending for a "top-down" look
    plot_data = top_10_states_amount.sort_values(ascending=True)
    plot_data.plot(kind='barh', color='#ff6b6b', ax=ax2)

    # Add labels and title
    ax2.set_title('Top 10 States by Total Insurance Transaction Amount', fontsize=16, pad=20)
    ax2.set_xlabel('Total Transaction Amount (in Billions)', fontsize=12)
    ax2.set_ylabel('State', fontsize=12)

    # Add data labels to the bars
    for index, value in enumerate(plot_data.values):
        ax2.text(value, index, f' {value:,.2f}B', va='center', ha='left')

    ax2.tick_params(axis='x', rotation=0)
    st.pyplot(fig2)

    st.markdown("---")

    # =================================================================
    # VIZ 3: Quarterly Transaction Count Trend
    # =================================================================
    st.header("3. Total Insurance Transaction Count Trend Over Quarters")

    df['TimeKey'] = df['Year'].astype(int) * 10 + df['Quarter'].astype(int)
    df['QuarterLabel'] = 'Q' + df['Quarter'].astype(str) + ' ' + df['Year'].astype(str)

    # Group by the chronological key and sum the transaction count
    quarterly_trend = df.groupby(['TimeKey', 'QuarterLabel'])['TransactionCount'].sum().reset_index()

    # Ensure it's sorted by the TimeKey
    quarterly_trend = quarterly_trend.sort_values(by='TimeKey')

    # Extract the final series for plotting
    plot_data = quarterly_trend.set_index('QuarterLabel')['TransactionCount']

    fig3, ax3 = plt.subplots(figsize=(14, 7))
    plt.style.use('seaborn-v0_8-darkgrid')  # Apply style temporarily

    # Plot the trend
    plot_data.plot(
        kind='line',
        marker='o',
        linestyle='-',
        color='#1a759f',
        linewidth=2,
        markersize=6,
        ax=ax3
    )

    # Add labels and title
    ax3.set_title('Total Insurance Transaction Count Trend Over Quarters', fontsize=16, pad=20)
    ax3.set_xlabel('Quarter and Year', fontsize=12)
    ax3.set_ylabel('Total Transaction Count (in Millions)', fontsize=12)

    # Improve x-axis readability by rotating labels and setting ticks
    ax3.tick_params(axis='x', rotation=45, which='major', labelsize=10)

    # Add a grid for better readability
    ax3.grid(True, linestyle='--', alpha=0.7)

    st.pyplot(fig3)

    st.markdown("---")

    # =================================================================
    # VIZ 4: District Trend in Top State (Multi-Line Chart) - NEW CODE
    # =================================================================
    st.header("4. Quarterly Transaction Trend for Top 5 Districts in Leading State")

    # 3a. Identify the Top State
    top_state = df.groupby('State')['TransactionCount'].sum().idxmax()
    st.info(f"The analysis focuses on **{top_state}** (the state with the highest total transaction count).")

    # 3b. Filter for Top State Districts
    df_top_state_districts = df[
        (df['State'] == top_state) &
        (df['EntityType'] == 'districts')
        ].copy()

    if df_top_state_districts.empty:
        st.warning(
            f"No district data (EntityType='districts') available for the top state: {top_state}. Skipping trend visualization.")
    else:
        # 3c. Identify Top 5 Districts in that state
        top_5_districts = df_top_state_districts.groupby('EntityName')['TransactionCount'].sum().nlargest(
            5).index.tolist()

        # 3d. Prepare data for trend plot
        df_trend = df_top_state_districts[df_top_state_districts['EntityName'].isin(top_5_districts)].copy()

        # Create chronological key and label
        df_trend['TimeKey'] = df_trend['Year'].astype(int) * 10 + df_trend['Quarter'].astype(int)
        df_trend['QuarterLabel'] = 'Q' + df_trend['Quarter'].astype(str) + ' ' + df_trend['Year'].astype(str)

        # Group for plotting
        trend_data = df_trend.groupby(['TimeKey', 'QuarterLabel', 'EntityName'])['TransactionCount'].sum().reset_index()

        # Sort data chronologically for clean plotting
        trend_data = trend_data.sort_values(by='TimeKey')

        fig4, ax4 = plt.subplots(figsize=(12, 7))
        plt.style.use('seaborn-v0_8-darkgrid')

        # Plot using seaborn for easy multi-line plotting
        sns.lineplot(
            data=trend_data,
            x='QuarterLabel',
            y='TransactionCount',
            hue='EntityName',
            marker='o',
            linewidth=2,
            ax=ax4
        )

        # Add labels and title
        ax4.set_title(
            f'Quarterly Transaction Count Trend for Top 5 Districts in {top_state}',
            fontsize=16,
            pad=20
        )
        ax4.set_xlabel('Quarter and Year', fontsize=12)
        ax4.set_ylabel('Total Transaction Count (in Millions)', fontsize=12)

        # Improve x-axis readability by rotating labels and setting ticks
        ax4.tick_params(axis='x', rotation=45, which='major', labelsize=10)

        # Move the legend outside the plot
        ax4.legend(
            title='District',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make space for the legend
        st.pyplot(fig4)

    st.markdown("---")
    plt.figure(figsize=(12, 6))
    trend_data.plot(kind='line', marker='o', linestyle='-', ax=plt.gca())
    plt.title(f'Transaction Count Trend for Top 5 Districts in {top_state}', fontsize=14)
    plt.xlabel('Quarter and Year')
    plt.ylabel('Total Transaction Count')
    # plt.xticks(
    #     ticks=np.arange(len(trend_data)),
    #     labels=trend_data.index.get_level_values('QuarterLabel'),
    #     rotation=45, ha='right'
    # )
    plt.legend(title='District', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
    # =================================================================
    # VIZ 5: Top 10 Districts by Transaction Count
    # =================================================================
    st.header("5. Top 10 Districts by Total Insurance Transaction Count")

    # Group and sum (using the mock district data generated above)
    df_districts = df[df['EntityType'] == 'districts']
    top_districts_count = df_districts.groupby('EntityName')['TransactionCount'].sum()
    top_10_districts_count = top_districts_count.nlargest(10)

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    plt.style.use('ggplot')

    # Plot
    top_10_districts_count.sort_values(ascending=True).plot(kind='barh', color='#1f77b4', ax=ax5)

    # Add labels to bars
    for index, value in enumerate(top_10_districts_count.sort_values(ascending=True).values):
        ax5.text(value, index, f'{value:,.2f}M', va='center', ha='left')

    ax5.set_title('Top 10 Districts by Total Insurance Transaction Count', fontsize=14)
    ax5.set_xlabel('Total Transaction Count (in Millions)')
    ax5.set_ylabel('District Name')
    st.pyplot(fig5)

    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(df)


def analyze_user_registration():
    """
    Performs user registration analysis and generates five key visualizations
    using Streamlit and Seaborn/Matplotlib.

    Args:
        df (pd.DataFrame): DataFrame containing the 'top_user' data.
    """
    st.set_page_config(layout="wide")
    st.title("User Registration Analysis Dashboard")
    st.markdown("---")
    df = GetData("top_user")
    # Ensure RegisteredUsers is treated as numeric
    df['RegisteredUsers'] = pd.to_numeric(df['RegisteredUsers'], errors='coerce')
    df.dropna(subset=['RegisteredUsers'], inplace=True)

    if df.empty:
        st.warning("The dataset is empty after cleaning. Cannot generate plots.")
        return

    # --- Analysis 1: Annual Registration Trend (Plot 1 - Line Plot) ---
    st.subheader("1. Annual User Registration Trend")
    yearly_data = df.groupby('Year')['RegisteredUsers'].sum().reset_index()

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=yearly_data, x='Year', y='RegisteredUsers', marker='o', color='darkblue', ax=ax1)

    ax1.set_title('Total Registered Users Over Time', fontsize=16)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Total Registered Users (Millions)', fontsize=12)
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.grid(axis='y', linestyle='--')
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # --- Analysis 2: State-wise Registration Share (Plot 2 - Bar Plot) ---
    st.subheader("2. Total Registered Users by State")
    state_data = df.groupby('State')['RegisteredUsers'].sum().sort_values(ascending=False).reset_index()

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=state_data, x='State', y='RegisteredUsers', palette='viridis', ax=ax2)

    ax2.set_title('Geographical Distribution of Registered Users', fontsize=16)
    ax2.set_xlabel('State', fontsize=12)
    ax2.set_ylabel('Total Registered Users (Millions)', fontsize=12)
    ax2.tick_params(axis='x', rotation=90)
    ax2.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # --- Analysis 3: Top 10 Entities by Users (Plot 3 - Horizontal Bar Plot) ---
    st.subheader("3. Top 10 Entities by Total Registered Users")
    entity_data = df.groupby('EntityName')['RegisteredUsers'].sum().nlargest(10).reset_index()
    # Sort descending for plotting from top to bottom
    entity_data_sorted = entity_data.sort_values(by='RegisteredUsers', ascending=False)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=entity_data_sorted,
        y='EntityName',
        x='RegisteredUsers',
        palette='magma',
        ax=ax3
    )

    ax3.set_title('Top 10 Performing Entities', fontsize=16)
    ax3.set_ylabel('Entity Name', fontsize=12)
    ax3.set_xlabel('Total Registered Users (Millions)', fontsize=12)
    ax3.ticklabel_format(style='plain', axis='x')
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # --- Analysis 4: Quarterly Registration Momentum (Plot 4 - Bar Plot) ---
    st.subheader("4. Quarterly Registration Momentum Across All Years")
    # Combine Year and Quarter for a continuous time axis
    df['TimePeriod'] = df['Year'].astype(str) + '-Q' + df['Quater'].astype(str)
    quarterly_data = df.groupby('TimePeriod')['RegisteredUsers'].sum().reset_index()

    # Sort the quarterly data correctly
    def sort_key(tp):
        year, quarter = tp.split('-Q')
        return int(year) * 100 + int(quarter)

    quarterly_data['SortKey'] = quarterly_data['TimePeriod'].apply(sort_key)
    quarterly_data = quarterly_data.sort_values('SortKey')

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=quarterly_data,
        x='TimePeriod',
        y='RegisteredUsers',
        palette='Greens_r',
        ax=ax4
    )

    ax4.set_title('Registration Momentum (Quarterly View)', fontsize=16)
    ax4.set_xlabel('Time Period (Year-Quarter)', fontsize=12)
    ax4.set_ylabel('Total Registered Users (Millions)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    # --- Analysis 5: Entity Type Distribution (Plot 5 - Horizontal Bar Chart) ---
    st.subheader("5. Distribution of Users by Entity Type (Comparison Plot)")
    entity_type_data = df.groupby('EntityType')['RegisteredUsers'].sum().reset_index()

    # Sort data for clean horizontal display (highest magnitude at the top)
    entity_type_data_sorted = entity_type_data.sort_values(by='RegisteredUsers', ascending=False)

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=entity_type_data_sorted,
        y='EntityType',
        x='RegisteredUsers',
        palette='cubehelix',
        ax=ax5
    )

    ax5.set_title('User Share by Entity Category', fontsize=16)
    ax5.set_ylabel('Entity Type', fontsize=12)
    ax5.set_xlabel('Total Registered Users (Millions)', fontsize=12)
    ax5.ticklabel_format(style='plain', axis='x')
    plt.tight_layout()

    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)



def analyze_market_transactions():
    """
    Performs transaction analysis for market expansion metrics and generates
    five key visualizations based on the 'map_map' data structure.

    Args:
        df (pd.DataFrame): DataFrame containing transaction data.
    """
    st.title("Transaction Analysis for Market Expansion Dashboard")
    st.markdown("---")
    df = GetData("map_map")
    # Data Cleaning and Preparation
    df.fillna(0, inplace=True)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)

    # Pre-calculate necessary metrics
    state_amount = df.groupby('State')['MetricAmount'].sum().sort_values(ascending=False)

    # Calculate type_amount needed for Plot 5
    type_amount = df.groupby('MetricType')['MetricAmount'].sum().sort_values(ascending=False)

    # --- 1. State-wise total transaction amount (Bar Plot) ---
    st.subheader("1. State-wise Total Transaction Amount")
    plt.style.use('ggplot')

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=state_amount.index, y=state_amount.values, palette='viridis', ax=ax1)

    ax1.set_title('Total Transaction Amount by State', fontsize=16)
    ax1.set_xlabel('State', fontsize=12)
    ax1.set_ylabel('Total Transaction Amount', fontsize=12)
    ax1.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # --- 2. Year-on-year total transaction count (Line Plot) ---
    st.subheader("2. Year-on-Year Total Transaction Count")
    year_count = df.groupby('Year')['MetricCount'].sum()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    year_count.plot(kind='line', marker='o', color='blue', ax=ax2)

    ax2.set_title('Year-on-Year Total Transaction Count', fontsize=16)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Total Transaction Count', fontsize=12)
    ax2.set_xticks(year_count.index)  # Ensure all years are displayed
    ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # --- 3. Top 5 districts by transaction amount in the latest year (Bar Plot) ---
    st.subheader("3. Top 5 Districts by Transaction Amount")
    latest_year = df['Year'].max()
    top_districts = df[df['Year'] == latest_year].groupby('DistrictName')['MetricAmount'].sum().nlargest(5)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_districts.index, y=top_districts.values, palette='magma', ax=ax3)

    ax3.set_title(f'Top 5 Districts by Total Transaction Amount in {latest_year}', fontsize=16)
    ax3.set_xlabel('District Name', fontsize=12)
    ax3.set_ylabel('Total Transaction Amount', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # --- 4. Average transaction amount by metric type and year (Line Plot) ---
    st.subheader("4. Average Transaction Amount (Ticket Size) by Metric Type and Year")
    avg_amount_by_year_type = df.groupby(['Year', 'MetricType'])['MetricAmount'].mean().reset_index()

    fig4, ax4 = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=avg_amount_by_year_type, x='Year', y='MetricAmount', hue='MetricType', marker='o', ax=ax4)

    ax4.set_title('Average Transaction Amount (Ticket Size) by Metric Type and Year', fontsize=16)
    ax4.set_xlabel('Year', fontsize=12)
    ax4.set_ylabel('Average Metric Amount', fontsize=12)
    ax4.set_xticks(avg_amount_by_year_type['Year'].unique())
    ax4.legend(title='Metric Type')
    ax4.grid(True, axis='y')
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    # --- 5. Quarterly trend for the leading metric in the top state (Line Plot) ---
    st.subheader("5. Quarterly Trend for Leading Metric in Top State")

    # Dynamic selection of State and MetricType
    major_state = state_amount.index[0]
    major_metric_type = type_amount.index[0]

    # Filter and aggregate
    quarterly_trend = df[(df['State'] == major_state) & (df['MetricType'] == major_metric_type)].copy()

    # Create a sortable quarter column (Year * 10 + Quarter) for proper line plot order
    quarterly_trend['Year_Quater'] = quarterly_trend['Year'] * 10 + quarterly_trend['Quater']
    quarterly_trend = quarterly_trend.sort_values('Year_Quater')

    # Group by the original 'Quater' for simpler labeling, but the x-axis will use a sequence
    quarterly_agg = quarterly_trend.groupby(['Year', 'Quater'])['MetricAmount'].sum().reset_index()
    quarterly_agg['Label'] = quarterly_agg['Year'].astype(str) + '-Q' + quarterly_agg['Quater'].astype(str)

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(quarterly_agg['Label'], quarterly_agg['MetricAmount'], marker='o', linestyle='-', color='red')

    ax5.set_title(f'Quarterly Trend for "{major_metric_type}" in {major_state}', fontsize=16)
    ax5.set_xlabel('Year-Quarter', fontsize=12)
    ax5.set_ylabel('Total Transaction Amount', fontsize=12)
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True)
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)


def analyze_user_engagement():
    """
    Performs user engagement and growth strategy analysis using five
    visualizations based on the 'map_user' data structure.

    Args:
        df (pd.DataFrame): DataFrame containing user registration and geographical data.
    """
    st.title("User Engagement and Growth Strategy Dashboard")
    st.markdown("---")
    df = GetData("map_user")
    # Data Cleaning and Preparation
    df.fillna(0, inplace=True)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)

    # Ensure RegisteredUsers is numeric and clean
    df['RegisteredUsers'] = pd.to_numeric(df['RegisteredUsers'], errors='coerce')
    df.dropna(subset=['RegisteredUsers'], inplace=True)

    if df.empty:
        st.warning("The dataset is empty after cleaning. Cannot generate plots.")
        return

    # --- Pre-calculations needed for multiple plots ---
    q1_data = df.groupby('State')['RegisteredUsers'].sum().sort_values(ascending=False).head(10)
    top_state = q1_data.index[0]
    latest_year = df['Year'].max()

    # --- 1. Top 10 States by Total Registered Users (Bar Plot) ---
    st.subheader("1. Top 10 States by Total Registered Users")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=q1_data.index, y=q1_data.values, palette='viridis', ax=ax1)
    # sns.barplot(data=state_volume, x='State', y='TransactionCount', palette='viridis', ax=ax2)

    ax1.set_title(f'Top 10 States by Total Registered Users ({df["Year"].min()}-{latest_year})', fontsize=14)
    ax1.set_xlabel('State', fontsize=12)
    ax1.set_ylabel('Total Registered Users', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # --- 2. Top Districts in the Top State (Bar Plot) ---
    st.subheader(f"2. Top 10 Districts in {top_state}")

    q2_data = df[df['State'] == top_state]
    q2_data = q2_data.groupby('DistrictName')['RegisteredUsers'].sum().sort_values(ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=q2_data.index, y=q2_data.values, palette='viridis', ax=ax2)

    ax2.set_title(f'Top 10 Districts in {top_state} by Total Registered Users', fontsize=14)
    ax2.set_xlabel('District Name', fontsize=12)
    ax2.set_ylabel('Total Registered Users', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # --- 3. Overall Yearly Growth (Line Plot) ---
    st.subheader("3. Overall Yearly Growth of Registered Users")
    q3_data = df.groupby('Year')['RegisteredUsers'].sum()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=q3_data.index, y=q3_data.values, marker='o', color='forestgreen', linewidth=2, ax=ax3)

    ax3.set_title('Overall Yearly Growth of Registered Users', fontsize=14)
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Total Registered Users', fontsize=12)
    ax3.set_xticks(q3_data.index)
    ax3.grid(axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # --- 4. Quarterly Seasonality (Latest Year) (Bar Plot) ---
    st.subheader(f"4. Quarterly Seasonality in {latest_year}")

    q4_data = df[df['Year'] == latest_year]
    q4_data = q4_data.groupby('Quater')['RegisteredUsers'].sum()

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=q4_data.index, y=q4_data.values, palette='RdYlGn', ax=ax4)

    ax4.set_title(f'Quarterly Seasonality in {latest_year}', fontsize=14)
    ax4.set_xlabel('Quarter', fontsize=12)
    ax4.set_ylabel('Total Registered Users', fontsize=12)
    ax4.set_xticks(q4_data.index)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    # --- 5. Top 5 States Quarterly Comparison (Latest Year) (Line Plot) ---
    st.subheader(f"5. Top 5 States Quarterly Comparison in {latest_year}")

    top_5_states = q1_data.head(5).index.tolist()

    q5_data = df[(df['Year'] == latest_year) & (df['State'].isin(top_5_states))]
    # Aggregate by Quarter and State, then pivot to get states as columns
    q5_data = q5_data.groupby(['Quater', 'State'])['RegisteredUsers'].sum().unstack()

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    q5_data.plot(kind='line', marker='o', linewidth=2,
                 ax=ax5)  # Using pandas plot on the grouped data for multi-line plot

    ax5.set_title(f'Top 5 States Quarterly Comparison in {latest_year}', fontsize=14)
    ax5.set_xlabel('Quarter', fontsize=12)
    ax5.set_ylabel('Total Registered Users', fontsize=12)
    ax5.set_xticks(q5_data.index)
    ax5.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)


def analyze_insurance_engagement():
    """
    Performs insurance engagement analysis and generates five key visualizations
    using Streamlit and Seaborn/Matplotlib based on the 'map_insurance' data.

    Args:
        df (pd.DataFrame): DataFrame containing insurance transaction data.
    """
    st.title("Insurance Engagement Analysis Dashboard")
    st.markdown("---")
    df = GetData("map_insurance")
    # Data Cleaning and Preparation
    df.fillna(0, inplace=True)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)

    # Ensure metric columns are correctly typed
    df['MetricCount'] = pd.to_numeric(df['MetricCount'], errors='coerce').fillna(0).astype(int)
    df['MetricAmount'] = pd.to_numeric(df['MetricAmount'], errors='coerce').fillna(0).astype(float)

    if df.empty:
        st.warning("The dataset is empty after cleaning. Cannot generate plots.")
        return

    # Filter out 'TOTAL' MetricType entries for type-specific analysis
    df_filtered = df[df['MetricType'] != 'TOTAL'].copy()
    latest_year = df['Year'].max()

    # --- 1. Top 5 States by Total Insurance Transaction Count (Latest Year) ---
    st.subheader(f"1. Top 5 States by Total Insurance Count in {latest_year}")

    # Use the full df for state-level aggregation
    df_latest_year = df[df['Year'] == latest_year]
    state_counts = df_latest_year.groupby('State')['MetricCount'].sum().nlargest(5).reset_index()

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=state_counts, x='State', y='MetricCount', palette='viridis', ax=ax1)

    ax1.set_title('Top 5 States by Total Insurance Count (Latest Year)', fontsize=14)
    ax1.set_xlabel('State', fontsize=12)
    ax1.set_ylabel('Total Transaction Count', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # --- 2. Total Insurance Transaction Amount Over Years (Line Plot) ---
    st.subheader("2. Total Insurance Transaction Amount Over Years")
    yearly_amount = df.groupby('Year')['MetricAmount'].sum()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=yearly_amount.index, y=yearly_amount.values, marker='o', linestyle='-', color='teal', ax=ax2)

    ax2.set_title('Total Insurance Transaction Amount Trend', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Total Transaction Amount', fontsize=12)
    ax2.set_xticks(yearly_amount.index)
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # --- 3. Average Transaction Amount (Ticket Size) per Quarter (Bar Plot) ---
    st.subheader("3. Average Transaction Amount (Ticket Size) per Quarter")

    # Calculate average amount for each quarter
    quarterly_avg_amount = df.groupby('Quater')['MetricAmount'].mean().reset_index()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=quarterly_avg_amount, x='Quater', y='MetricAmount', palette='YlOrBr', ax=ax3)

    ax3.set_title('Average Insurance Transaction Amount by Quarter (All Years)', fontsize=14)
    ax3.set_xlabel('Quarter', fontsize=12)
    ax3.set_ylabel('Average Transaction Amount', fontsize=12)
    ax3.set_xticks(quarterly_avg_amount['Quater'])
    ax3.ticklabel_format(style='plain', axis='y')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # --- 4. Quarterly Count Distribution (Bar Plot) ---
    st.subheader("4. Total Insurance Count Distribution by Quarter")
    quarterly_counts = df.groupby('Quater')['MetricCount'].sum().reset_index()

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=quarterly_counts, x='Quater', y='MetricCount', palette='viridis', ax=ax4)

    ax4.set_title('Total Insurance Count Distribution by Quarter (All Years)', fontsize=14)
    ax4.set_xlabel('Quarter', fontsize=12)
    ax4.set_ylabel('Total Transaction Count', fontsize=12)
    ax4.set_xticks(quarterly_counts['Quater'])
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    # --- 5. Top 5 Districts by Total Insurance Transaction Amount (Bar Plot) ---
    st.subheader("5. Top 5 Districts by Total Insurance Transaction Amount")

    district_amounts = df.groupby('DistrictName')['MetricAmount'].sum().nlargest(5).reset_index()

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=district_amounts, x='DistrictName', y='MetricAmount', palette='viridis', ax=ax5)

    ax5.set_title('Top 5 Districts by Total Insurance Transaction Amount (All Years)', fontsize=14)
    ax5.set_xlabel('District Name', fontsize=12)
    ax5.set_ylabel('Total Transaction Amount', fontsize=12)
    ax5.tick_params(axis='x', rotation=45)
    ax5.ticklabel_format(style='plain', axis='y')
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)

def main():
    st.set_page_config(
        page_title="📊 PhonePay Visualization",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Store the main selection in session state so other functions can access it
    st.session_state['main_selection'] = st.sidebar.radio(
        "Topics:",
        ["Home", "Aggregated Data Visualization", "Map Data Visualization", "Top Data Visualization"],
        key='main_radio_selection'
    )

    menu_options = {
        "Home": home_page,
        "Aggregated Data Visualization": data_visualization_page,  # All aggregated/map/top pages use this handler
        "Map Data Visualization": data_visualization_page,
        "Top Data Visualization": data_visualization_page,
    }

    st.sidebar.markdown("---")


    page_function = menu_options[st.session_state['main_selection']]
    page_function()


# Execute the main function
if __name__ == "__main__":
    main()
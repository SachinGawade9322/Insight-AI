# # import pandas as pd
# # import numpy as np
# # from datetime import datetime
# # import matplotlib.pyplot as plt
# # from typing import Dict, List, Optional, Union
# # from sklearn.linear_model import LinearRegression


# # # Configuration
# # pd.set_option('display.max_columns', None)
# # pd.set_option('display.width', 1000)

# # class TransactionAnalyzer:
# #     def __init__(self, data_path: str = None, df: pd.DataFrame = None):
# #         """
# #         Initialize the TransactionAnalyzer with robust date handling.
        
# #         Args:
# #             data_path: Path to CSV file containing transaction data
# #             df: Pre-loaded DataFrame containing transaction data
# #         """
# #         if data_path:
# #             self.df = self._load_data(data_path)
# #         elif df is not None:
# #             self.df = df.copy()
# #         else:
# #             raise ValueError("Either data_path or df must be provided")
            
# #         # Initialize category mappings before preparing data
# #         self._initialize_mappings()
# #         self._prepare_data()
    
# #     def _initialize_mappings(self):
# #         """Initialize all category and company mappings."""
# #         self.categories = {
# #             "Deposit & Addition": ["Salary", "Bonus", "Interest Earned"],
# #             "Bills & Utilities": ["Electricity", "Water", "Internet", "Gas"],
# #             "Shopping & Retail": ["Clothing", "Electronics", "Groceries"],
# #             "Transport": ["Fuel", "Taxi", "Public Transport"],
# #             "Entertainment": ["Movies", "Concerts", "Games"],
# #             "Healthcare": ["Pharmacy", "Doctor", "Hospital Bills"],
# #             "Food & Dining": ["Restaurants", "Fast Food", "Coffee Shops"],
# #             "Travel": ["Flights", "Hotels", "Car Rentals"],
# #             "Subscription": ["Netflix", "Spotify", "Gym Membership"]
# #         }
        
# #         self.company_names = {
# #             "Salary": ["Google", "Microsoft", "Amazon"],
# #             "Bonus": ["Tesla", "Meta", "Apple"],
# #             "Interest Earned": ["Bank of America", "Chase", "Wells Fargo"],
# #             "Electricity": ["Duke Energy", "Pacific Gas & Electric"],
# #             "Water": ["City Water Corp"],
# #             "Internet": ["Comcast", "Verizon", "AT&T"],
# #             "Gas": ["Shell", "BP", "ExxonMobil"],
# #             "Clothing": ["Nike", "Adidas", "H&M"],
# #             "Electronics": ["Best Buy", "Apple", "Samsung"],
# #             "Groceries": ["Walmart", "Costco", "Target"],
# #             "Fuel": ["Shell", "Chevron", "BP"],
# #             "Taxi": ["Uber", "Lyft"],
# #             "Public Transport": ["MTA", "BART"],
# #             "Movies": ["AMC Theatres", "Cinemark"],
# #             "Concerts": ["Ticketmaster", "Live Nation"],
# #             "Games": ["Steam", "PlayStation Store"],
# #             "Pharmacy": ["CVS", "Walgreens"],
# #             "Doctor": ["Kaiser Permanente", "Mayo Clinic"],
# #             "Hospital Bills": ["Cedars-Sinai", "Mount Sinai"],
# #             "Restaurants": ["McDonald's", "KFC", "Subway"],
# #             "Fast Food": ["Domino's", "Burger King"],
# #             "Coffee Shops": ["Starbucks", "Dunkin' Donuts"],
# #             "Flights": ["American Airlines", "Delta"],
# #             "Hotels": ["Hilton", "Marriott"],
# #             "Car Rentals": ["Hertz", "Enterprise"],
# #             "Netflix": ["Netflix Inc."],
# #             "Spotify": ["Spotify Ltd."],
# #             "Gym Membership": ["Gold's Gym", "Planet Fitness"]
# #         }
        
# #         # Build reverse mapping
# #         self.subcategory_to_main = {}
# #         for main_cat, subcats in self.categories.items():
# #             for subcat in subcats:
# #                 self.subcategory_to_main[subcat] = main_cat
    
# #     def _load_data(self, data_path: str) -> pd.DataFrame:
# #         """Load transaction data from CSV file with date parsing."""
# #         # Try reading with different date formats
# #         try:
# #             return pd.read_csv(data_path, parse_dates=['Date'], dayfirst=True)
# #         except:
# #             try:
# #                 return pd.read_csv(data_path, parse_dates=['Date'], format='%m/%d/%Y')
# #             except:
# #                 return pd.read_csv(data_path, parse_dates=['Date'], infer_datetime_format=True)
    
# #     def _prepare_data(self):
# #         """Clean and prepare the transaction data with robust date handling."""
# #         # Handle date parsing with multiple attempts
# #         if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
# #             try:
# #                 self.df['Date'] = pd.to_datetime(self.df['Date'], format='%m/%d/%Y')
# #             except ValueError:
# #                 try:
# #                     self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
# #                 except ValueError:
# #                     try:
# #                         self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d')
# #                     except ValueError:
# #                         self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
# #         # Check for any remaining NaT values
# #         if self.df['Date'].isna().any():
# #             print(f"Warning: {self.df['Date'].isna().sum()} dates couldn't be parsed and were set to NaT")
# #             print("Sample of problematic rows:")
# #             print(self.df[self.df['Date'].isna()].head())
        
# #         # Extract datetime features
# #         self.df['Year'] = self.df['Date'].dt.year
# #         self.df['Month'] = self.df['Date'].dt.month
# #         self.df['Day'] = self.df['Date'].dt.day
# #         self.df['Weekday'] = self.df['Date'].dt.day_name()
# #         self.df['Quarter'] = self.df['Date'].dt.quarter
        
# #         # Ensure Amount is numeric
# #         self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
        
# #         # Create income/expense flag
# #         self.df['Type'] = np.where(self.df['Amount'] >= 0, 'Income', 'Expense')
        
# #         # Fill missing categories
# #         self._fill_missing_categories()
    
# #     def _fill_missing_categories(self):
# #         """Fill missing categories/subcategories based on company names."""
# #         # Create mapping from company to subcategory
# #         company_to_subcat = {}
# #         for subcat, companies in self.company_names.items():
# #             for company in companies:
# #                 company_to_subcat[company] = subcat
        
# #         # Fill missing subcategories
# #         mask = self.df['Subcategory'].isna()
# #         self.df.loc[mask, 'Subcategory'] = self.df.loc[mask, 'Company'].map(company_to_subcat)
        
# #         # Fill missing main categories
# #         mask = self.df['Main Category'].isna()
# #         self.df.loc[mask, 'Main Category'] = self.df.loc[mask, 'Subcategory'].map(self.subcategory_to_main)
    
# #     # --------------------------
# #     # Basic Summary Queries (1-10)
# #     # --------------------------
    
# #     def total_income(self) -> float:
# #         """Calculate total income."""
# #         return self.df[self.df['Type'] == 'Income']['Amount'].sum()
    
# #     def total_expenses(self) -> float:
# #         """Calculate total expenses."""
# #         return self.df[self.df['Type'] == 'Expense']['Amount'].sum()
    
# #     def net_balance(self) -> float:
# #         """Calculate net balance (income - expenses)."""
# #         return self.df['Amount'].sum()
    
# #     def average_monthly_income(self) -> float:
# #         """Calculate average monthly income."""
# #         monthly_income = self.df[self.df['Type'] == 'Income'].groupby(
# #             [self.df['Date'].dt.year, self.df['Date'].dt.month]
# #         )['Amount'].sum()
# #         return monthly_income.mean()
    
# #     def average_monthly_expenses(self) -> float:
# #         """Calculate average monthly expenses."""
# #         monthly_expenses = self.df[self.df['Type'] == 'Expense'].groupby(
# #             [self.df['Date'].dt.year, self.df['Date'].dt.month]
# #         )['Amount'].sum()
# #         return monthly_expenses.mean()
    
# #     def income_by_category(self) -> pd.DataFrame:
# #         """Breakdown income by main category."""
# #         return self.df[self.df['Type'] == 'Income'].groupby('Main Category')['Amount'].sum()
    
# #     def expenses_by_category(self) -> pd.DataFrame:
# #         """Breakdown expenses by main category."""
# #         return self.df[self.df['Type'] == 'Expense'].groupby('Main Category')['Amount'].sum()
    
# #     def top_income_sources(self, n: int = 5) -> pd.DataFrame:
# #         """Identify top n income sources by subcategory."""
# #         return (self.df[self.df['Type'] == 'Income']
# #                 .groupby('Subcategory')['Amount']
# #                 .sum()
# #                 .nlargest(n))
    
# #     def top_expense_categories(self, n: int = 5) -> pd.DataFrame:
# #         """Identify top n expense categories by subcategory."""
# #         return (self.df[self.df['Type'] == 'Expense']
# #                 .groupby('Subcategory')['Amount']
# #                 .sum()
# #                 .nlargest(n))
    
# #     def transaction_counts(self) -> Dict[str, int]:
# #         """Count of income vs expense transactions."""
# #         return self.df['Type'].value_counts().to_dict()
    
# #     # --------------------------
# #     # Time-Based Analysis (11-20)
# #     # --------------------------
    
# #     def monthly_income_expense(self) -> pd.DataFrame:
# #         """Monthly income and expense summary."""
# #         return self.df.groupby(
# #             [self.df['Date'].dt.to_period('M'), 'Type']
# #         )['Amount'].sum().unstack()
    
# #     def yearly_summary(self) -> pd.DataFrame:
# #         """Yearly income, expense, and net balance."""
# #         yearly = self.df.groupby(self.df['Date'].dt.year)['Amount'].agg(['sum', 'mean', 'count'])
# #         yearly.columns = ['Net Balance', 'Average Transaction', 'Transaction Count']
# #         return yearly
    
# #     def quarterly_breakdown(self) -> pd.DataFrame:
# #         """Quarterly income and expense breakdown."""
# #         return self.df.groupby(
# #             [self.df['Date'].dt.year, 'Quarter', 'Type']
# #         )['Amount'].sum().unstack()
    
# #     def weekday_analysis(self) -> pd.DataFrame:
# #         """Spending patterns by day of week."""
# #         return self.df.groupby(['Weekday', 'Type'])['Amount'].sum().unstack()
    
# #     def monthly_trends(self, category: str = None) -> pd.DataFrame:
# #         """
# #         Monthly trends for income/expenses or specific category.
        
# #         Args:
# #             category: Optional main category to filter by
# #         """
# #         df = self.df
# #         if category:
# #             df = df[df['Main Category'] == category]
# #         return df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum()
    
# #     def year_over_year_growth(self) -> pd.DataFrame:
# #         """Year-over-year growth percentage for income and expenses."""
# #         yearly = self.df.groupby([self.df['Date'].dt.year, 'Type'])['Amount'].sum().unstack()
# #         return yearly.pct_change() * 100
    
# #     def busiest_spending_days(self, n: int = 10) -> pd.DataFrame:
# #         """Days with highest total spending."""
# #         return (self.df[self.df['Type'] == 'Expense']
# #                 .groupby('Date')['Amount']
# #                 .sum()
# #                 .nlargest(n))
    
# #     def seasonal_spending_patterns(self) -> pd.DataFrame:
# #         """Seasonal spending patterns by month."""
# #         return (self.df[self.df['Type'] == 'Expense']
# #                 .groupby(self.df['Date'].dt.month)['Amount']
# #                 .sum())
    
# #     def compare_periods(self, start1: str, end1: str, start2: str, end2: str) -> pd.DataFrame:
# #         """
# #         Compare spending between two time periods.
        
# #         Args:
# #             start1, end1: First period start and end dates (YYYY-MM-DD)
# #             start2, end2: Second period start and end dates (YYYY-MM-DD)
# #         """
# #         period1 = self.df[self.df['Date'].between(start1, end1)]
# #         period2 = self.df[self.df['Date'].between(start2, end2)]
        
# #         result = pd.DataFrame({
# #             'Period 1': period1.groupby('Main Category')['Amount'].sum(),
# #             'Period 2': period2.groupby('Main Category')['Amount'].sum()
# #         })
# #         result['Change'] = result['Period 2'] - result['Period 1']
# #         result['Change %'] = (result['Change'] / result['Period 1'].abs()) * 100
# #         return result
    
# #     def rolling_monthly_spending(self, window: int = 3) -> pd.Series:
# #         """Rolling average of monthly spending."""
# #         monthly = self.df[self.df['Type'] == 'Expense'].groupby(
# #             self.df['Date'].dt.to_period('M')
# #         )['Amount'].sum()
# #         return monthly.rolling(window=window).mean()
    
# #     # --------------------------
# #     # Category Analysis (21-30)
# #     # --------------------------
    
# #     def category_spending_distribution(self) -> pd.Series:
# #         """Percentage distribution of spending by category."""
# #         spending = self.df[self.df['Type'] == 'Expense'].groupby('Main Category')['Amount'].sum().abs()
# #         return (spending / spending.sum()) * 100
    
# #     def subcategory_spending_distribution(self, main_category: str = None) -> pd.Series:
# #         """
# #         Percentage distribution of spending by subcategory.
        
# #         Args:
# #             main_category: Optional main category to filter by
# #         """
# #         df = self.df[self.df['Type'] == 'Expense']
# #         if main_category:
# #             df = df[df['Main Category'] == main_category]
# #         spending = df.groupby('Subcategory')['Amount'].sum().abs()
# #         return (spending / spending.sum()) * 100
    
# #     def category_spending_over_time(self, category: str) -> pd.DataFrame:
# #         """Monthly spending for a specific category over time."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Main Category'] == category)]
# #                 .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                 .sum())
    
# #     def compare_categories(self, categories: List[str]) -> pd.DataFrame:
# #         """Compare spending between multiple categories over time."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Main Category'].isin(categories))]
# #                 .groupby([self.df['Date'].dt.to_period('M'), 'Main Category'])['Amount']
# #                 .sum()
# #                 .unstack())
    
# #     def top_companies_by_category(self, category: str, n: int = 5) -> pd.Series:
# #         """Top companies by spending in a specific category."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Main Category'] == category)]
# #                 .groupby('Company')['Amount']
# #                 .sum()
# #                 .nsmallest(n)
# #                 .abs())
    
# #     def category_spending_by_demographic(self, demographic: str = 'Weekday') -> pd.DataFrame:
# #         """
# #         Analyze category spending by demographic (weekday, month, etc.)
        
# #         Args:
# #             demographic: Column to group by ('Weekday', 'Month', 'Quarter')
# #         """
# #         return (self.df[self.df['Type'] == 'Expense']
# #                 .groupby([demographic, 'Main Category'])['Amount']
# #                 .sum()
# #                 .unstack())
    
# #     def expense_income_ratio_by_category(self) -> pd.DataFrame:
# #         """Ratio of expenses to income by category."""
# #         expenses = self.df[self.df['Type'] == 'Expense'].groupby('Main Category')['Amount'].sum().abs()
# #         income = self.df[self.df['Type'] == 'Income'].groupby('Main Category')['Amount'].sum()
# #         return (expenses / income) * 100
    
# #     def identify_high_spend_categories(self, threshold: float = 1000) -> pd.Series:
# #         """Identify categories where average transaction exceeds threshold."""
# #         return (self.df[self.df['Type'] == 'Expense']
# #                 .groupby('Main Category')['Amount']
# #                 .mean()
# #                 .loc[lambda x: x.abs() > threshold])
    
# #     def category_spending_variability(self) -> pd.DataFrame:
# #         """Variability (std dev) of spending by category."""
# #         return self.df[self.df['Type'] == 'Expense'].groupby('Main Category')['Amount'].std().abs()
    
# #     def category_spending_seasonality(self, category: str) -> pd.DataFrame:
# #         """Monthly seasonality for a specific category."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Main Category'] == category)]
# #                 .groupby(self.df['Date'].dt.month)['Amount']
# #                 .mean()
# #                 .abs())
    
# #     # --------------------------
# #     # Company/Subcategory Analysis (31-40)
# #     # --------------------------
    
# #     def top_spending_companies(self, n: int = 10) -> pd.Series:
# #         """Companies with highest total spending."""
# #         return (self.df[self.df['Type'] == 'Expense']
# #                 .groupby('Company')['Amount']
# #                 .sum()
# #                 .nsmallest(n)
# #                 .abs())
    
# #     def company_spending_over_time(self, company: str) -> pd.Series:
# #         """Monthly spending with a specific company."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Company'] == company)]
# #                 .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                 .sum())
    
# #     def subcategory_spending_trends(self, subcategory: str) -> pd.Series:
# #         """Monthly spending trends for a specific subcategory."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Subcategory'] == subcategory)]
# #                 .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                 .sum())
    
# #     def compare_companies(self, companies: List[str]) -> pd.DataFrame:
# #         """Compare spending between multiple companies over time."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Company'].isin(companies))]
# #                 .groupby([self.df['Date'].dt.to_period('M'), 'Company'])['Amount']
# #                 .sum()
# #                 .unstack())
    
# #     def company_frequency(self, n: int = 10) -> pd.Series:
# #         """Most frequent companies (by transaction count)."""
# #         return self.df['Company'].value_counts().head(n)
    
# #     def average_transaction_by_company(self) -> pd.Series:
# #         """Average transaction amount by company."""
# #         return self.df.groupby('Company')['Amount'].mean()
    
# #     def company_spending_by_category(self, company: str) -> pd.Series:
# #         """Breakdown of spending by category for a specific company."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Company'] == company)]
# #                 .groupby('Main Category')['Amount']
# #                 .sum()
# #                 .abs())
    
# #     def identify_recurring_expenses(self, frequency_threshold: int = 3) -> pd.Series:
# #         """Identify recurring expenses (same company/amount appearing multiple times)."""
# #         recurring = (self.df[self.df['Type'] == 'Expense']
# #                     .groupby(['Company', 'Amount'])
# #                     .size()
# #                     .reset_index(name='Count'))
# #         return recurring[recurring['Count'] >= frequency_threshold].sort_values('Count', ascending=False)
    
# #     def company_spending_distribution(self) -> pd.Series:
# #         """Percentage of total spending by company."""
# #         spending = self.df[self.df['Type'] == 'Expense'].groupby('Company')['Amount'].sum().abs()
# #         return (spending / spending.sum()) * 100
    
# #     def top_companies_by_subcategory(self, subcategory: str, n: int = 5) -> pd.Series:
# #         """Top companies by spending in a specific subcategory."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Subcategory'] == subcategory)]
# #                 .groupby('Company')['Amount']
# #                 .sum()
# #                 .nsmallest(n)
# #                 .abs())
    
# #     # --------------------------
# #     # Subscription Analysis (41-50)
# #     # --------------------------
    
# #     def subscription_expenses(self) -> pd.Series:
# #         """Total subscription expenses by service."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Main Category'] == 'Subscription')]
# #                 .groupby('Subcategory')['Amount']
# #                 .sum()
# #                 .abs())
    
# #     def monthly_subscription_costs(self) -> pd.DataFrame:
# #         """Monthly costs for each subscription service."""
# #         return ((self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Main Category'] == 'Subscription')]
# #                 .groupby([self.df['Date'].dt.to_period('M'), 'Subcategory'])['Amount']
# #                 .sum()
# #                 .unstack())
# #                 .abs())
    
# #     def subscription_cost_changes(self) -> pd.DataFrame:
# #         """Detect changes in subscription costs."""
# #         subs = self.df[(self.df['Type'] == 'Expense') & 
# #                       (self.df['Main Category'] == 'Subscription')]
        
# #         # Get unique amounts paid to each subscription service
# #         cost_changes = (subs.groupby(['Subcategory', 'Amount'])
# #                        .size()
# #                        .reset_index(name='Count')
# #                        .sort_values(['Subcategory', 'Amount']))
        
# #         # Identify services with multiple payment amounts
# #         return cost_changes[cost_changes['Subcategory'].duplicated(keep=False)]
    
# #     def subscription_renewal_dates(self) -> pd.DataFrame:
# #         """Estimate subscription renewal dates based on frequency."""
# #         subs = self.df[(self.df['Type'] == 'Expense') & 
# #                       (self.df['Main Category'] == 'Subscription')]
        
# #         renewal_dates = (subs.groupby('Subcategory')['Date']
# #                         .agg(['min', 'max', 'count'])
# #                         .rename(columns={'min': 'First Payment', 'max': 'Last Payment', 'count': 'Payments'}))
        
# #         renewal_dates['Avg Days Between'] = ((renewal_dates['Last Payment'] - renewal_dates['First Payment']).dt.days / 
# #                                             (renewal_dates['Payments'] - 1))
        
# #         renewal_dates['Next Expected'] = renewal_dates['Last Payment'] + pd.to_timedelta(
# #             renewal_dates['Avg Days Between'], unit='D')
        
# #         return renewal_dates
    
# #     def total_monthly_subscriptions(self) -> float:
# #         """Calculate total monthly subscription costs."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                       (self.df['Main Category'] == 'Subscription')]['Amount']
# #                 .sum()
# #                 .abs())
    
# #     def subscription_spending_over_time(self) -> pd.DataFrame:
# #         """Monthly subscription spending trends."""
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                       (self.df['Main Category'] == 'Subscription')]
# #                 .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                 .sum()
# #                 .abs())
    
# #     def subscription_cancellations(self) -> pd.DataFrame:
# #         """Identify potentially cancelled subscriptions (no recent payments)."""
# #         subs = self.df[(self.df['Type'] == 'Expense') & 
# #                       (self.df['Main Category'] == 'Subscription')]
        
# #         last_payments = subs.groupby('Subcategory')['Date'].max()
# #         cutoff_date = self.df['Date'].max() - pd.Timedelta(days=90)  # 3 months threshold
        
# #         return last_payments[last_payments < cutoff_date].sort_values()
    
# #     def subscription_cost_per_use(self, usage_data: Dict[str, int]) -> pd.DataFrame:
# #         """
# #         Calculate cost per use for subscriptions with usage data.
        
# #         Args:
# #             usage_data: Dictionary of {subcategory: usage_count}
# #         """
# #         total_costs = (self.df[(self.df['Type'] == 'Expense') & 
# #                              (self.df['Main Category'] == 'Subscription')]
# #                       .groupby('Subcategory')['Amount']
# #                       .sum()
# #                       .abs())
        
# #         cost_per_use = pd.DataFrame({
# #             'Total Cost': total_costs,
# #             'Usage': pd.Series(usage_data),
# #             'Cost Per Use': total_costs / pd.Series(usage_data)
# #         })
        
# #         return cost_per_use.dropna()
    
# #     def subscription_savings_opportunities(self, threshold: float = 10.0) -> pd.Series:
# #         """Identify subscription services with high monthly costs."""
# #         monthly_costs = (self.df[(self.df['Type'] == 'Expense') & 
# #                                (self.df['Main Category'] == 'Subscription')]
# #                         .groupby('Subcategory')['Amount']
# #                         .mean()
# #                         .abs())
        
# #         return monthly_costs[monthly_costs > threshold].sort_values(ascending=False)
    
# #     def subscription_usage_efficiency(self, usage_data: Dict[str, int]) -> pd.DataFrame:
# #         """
# #         Analyze subscription usage efficiency (cost vs usage).
        
# #         Args:
# #             usage_data: Dictionary of {subcategory: usage_count}
# #         """
# #         subs = self.df[(self.df['Type'] == 'Expense') & 
# #                       (self.df['Main Category'] == 'Subscription')]
        
# #         # Calculate total payments and payment frequency
# #         sub_info = subs.groupby('Subcategory').agg({
# #             'Amount': ['sum', 'count'],
# #             'Date': ['min', 'max']
# #         })
# #         sub_info.columns = ['Total Cost', 'Payment Count', 'First Payment', 'Last Payment']
# #         sub_info['Total Cost'] = sub_info['Total Cost'].abs()
        
# #         # Add usage data
# #         sub_info['Usage'] = pd.Series(usage_data)
        
# #         # Calculate metrics
# #         sub_info['Cost Per Use'] = sub_info['Total Cost'] / sub_info['Usage']
# #         sub_info['Usage Per Month'] = sub_info['Usage'] / (
# #             (sub_info['Last Payment'] - sub_info['First Payment']).dt.days / 30)
        
# #         return sub_info.dropna()
    
# #     # --------------------------
# #     # Anomaly Detection (51-60)
# #     # --------------------------
    
# #     def detect_unusual_spending(self, threshold: float = 2.0) -> pd.DataFrame:
# #         """Detect spending transactions that are unusually large."""
# #         expenses = self.df[self.df['Type'] == 'Expense']
        
# #         # Calculate z-scores for amounts (absolute value)
# #         amounts = expenses['Amount'].abs()
# #         z_scores = (amounts - amounts.mean()) / amounts.std()
        
# #         # Return unusual transactions
# #         unusual = expenses[z_scores.abs() > threshold].copy()
# #         unusual['Z-Score'] = z_scores[z_scores.abs() > threshold]
# #         return unusual.sort_values('Z-Score', ascending=False)
    
# #     def detect_income_anomalies(self, threshold: float = 2.0) -> pd.DataFrame:
# #         """Detect income transactions that are unusually large."""
# #         income = self.df[self.df['Type'] == 'Income']
        
# #         # Calculate z-scores for amounts
# #         z_scores = (income['Amount'] - income['Amount'].mean()) / income['Amount'].std()
        
# #         # Return unusual transactions
# #         unusual = income[z_scores.abs() > threshold].copy()
# #         unusual['Z-Score'] = z_scores[z_scores.abs() > threshold]
# #         return unusual.sort_values('Z-Score', ascending=False)
    
# #     def detect_spending_pattern_changes(self, window: int = 3, threshold: float = 0.3) -> pd.DataFrame:
# #         """
# #         Detect significant changes in spending patterns.
        
# #         Args:
# #             window: Number of months to compare
# #             threshold: Percentage change considered significant
# #         """
# #         monthly = (self.df[self.df['Type'] == 'Expense']
# #                   .groupby([self.df['Date'].dt.to_period('M'), 'Main Category'])['Amount']
# #                   .sum()
# #                   .unstack()
# #                   .abs())
        
# #         # Calculate rolling average and percentage change
# #         rolling_avg = monthly.rolling(window=window).mean()
# #         pct_change = monthly.pct_change(periods=window)
        
# #         # Find significant changes
# #         significant_changes = pct_change[pct_change.abs() > threshold].stack().reset_index()
# #         significant_changes.columns = ['Month', 'Category', 'Percentage Change']
        
# #         # Add context (previous average)
# #         prev_avg = rolling_avg.shift(window).stack().reset_index(name='Previous Average')
# #         significant_changes = significant_changes.merge(prev_avg, on=['Month', 'Category'])
        
# #         return significant_changes.dropna().sort_values('Percentage Change', ascending=False)
    
# #     def detect_missing_payments(self, expected_payments: Dict[str, int]) -> pd.DataFrame:
# #         """
# #         Detect missing expected recurring payments.
        
# #         Args:
# #             expected_payments: Dictionary of {company: expected_count}
# #         """
# #         actual_counts = self.df['Company'].value_counts()
# #         expected_counts = pd.Series(expected_payments)
        
# #         comparison = pd.DataFrame({
# #             'Expected': expected_counts,
# #             'Actual': actual_counts,
# #             'Difference': actual_counts - expected_counts
# #         }).dropna()
        
# #         return comparison[comparison['Difference'] < 0]
    
# #     def detect_duplicate_transactions(self, time_window: str = '1D') -> pd.DataFrame:
# #         """
# #         Detect potential duplicate transactions (same amount to same company within time window).
        
# #         Args:
# #             time_window: Time window to consider duplicates (e.g., '1D' for 1 day)
# #         """
# #         # Sort by company, amount, and date
# #         sorted_df = self.df.sort_values(['Company', 'Amount', 'Date'])
        
# #         # Calculate time difference between consecutive similar transactions
# #         sorted_df['Time Diff'] = sorted_df.groupby(['Company', 'Amount'])['Date'].diff()
        
# #         # Filter for duplicates within the time window
# #         duplicates = sorted_df[
# #             (sorted_df['Time Diff'] < pd.Timedelta(time_window)) & 
# #             (sorted_df['Time Diff'] > pd.Timedelta('0D'))
# #         ]
        
# #         return duplicates.sort_values(['Company', 'Amount', 'Date'])
    
# #     def identify_irregular_income(self, expected_frequency: str = 'M', threshold: float = 0.5) -> pd.DataFrame:
# #         """
# #         Identify irregular income patterns.
        
# #         Args:
# #             expected_frequency: Expected frequency ('D', 'W', 'M', 'Q', 'Y')
# #             threshold: Allowed deviation from expected amount
# #         """
# #         income = self.df[self.df['Type'] == 'Income']
        
# #         # Group by expected frequency
# #         freq_groups = income.groupby([pd.Grouper(key='Date', freq=expected_frequency), 'Subcategory'])
# #         freq_stats = freq_groups['Amount'].agg(['mean', 'std']).reset_index()
        
# #         # Identify irregular payments (deviating from mean)
# #         irregular = income.merge(freq_stats, on=['Date', 'Subcategory'])
# #         irregular['Deviation'] = (irregular['Amount'] - irregular['mean']).abs()
        
# #         return irregular[irregular['Deviation'] > (threshold * irregular['mean'])].sort_values('Deviation', ascending=False)
    
# #     def detect_category_spending_outliers(self, category: str, threshold: float = 2.0) -> pd.DataFrame:
# #         """
# #         Detect outlier transactions within a specific category.
        
# #         Args:
# #             category: Main category to analyze
# #             threshold: Z-score threshold for outliers
# #         """
# #         category_expenses = self.df[
# #             (self.df['Type'] == 'Expense') & 
# #             (self.df['Main Category'] == category)
# #         ]
        
# #         if category_expenses.empty:
# #             return pd.DataFrame()
        
# #         # Calculate z-scores
# #         amounts = category_expenses['Amount'].abs()
# #         z_scores = (amounts - amounts.mean()) / amounts.std()
        
# #         # Return outliers
# #         outliers = category_expenses[z_scores.abs() > threshold].copy()
# #         outliers['Z-Score'] = z_scores[z_scores.abs() > threshold]
# #         return outliers.sort_values('Z-Score', ascending=False)
    
# #     def identify_spending_spikes(self, window: int = 3, threshold: float = 1.5) -> pd.DataFrame:
# #         """
# #         Identify sudden spikes in spending.
        
# #         Args:
# #             window: Rolling window size in months
# #             threshold: Multiple of rolling average considered a spike
# #         """
# #         monthly = (self.df[self.df['Type'] == 'Expense']
# #                   .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                   .sum()
# #                   .abs())
        
# #         rolling_avg = monthly.rolling(window=window).mean()
# #         spikes = monthly[monthly > (rolling_avg * threshold)]
        
# #         return spikes.dropna().to_frame('Spike Amount')
    
# #     def detect_unusual_cashflow(self, window: int = 3, threshold: float = 0.3) -> pd.DataFrame:
# #         """
# #         Detect unusual changes in cash flow (income vs expenses).
        
# #         Args:
# #             window: Rolling window size in months
# #             threshold: Percentage change considered unusual
# #         """
# #         monthly = self.df.groupby(self.df['Date'].dt.to_period('M'))['Amount'].sum()
# #         pct_change = monthly.pct_change()
        
# #         unusual = pct_change[pct_change.abs() > threshold].to_frame('Percentage Change')
# #         unusual['Previous'] = monthly.shift(1)[unusual.index]
# #         unusual['Current'] = monthly[unusual.index]
        
# #         return unusual.sort_values('Percentage Change', ascending=False)
    
# #     def identify_changed_habits(self, baseline_start: str, baseline_end: str, 
# #                               comparison_start: str, comparison_end: str, 
# #                               threshold: float = 0.2) -> pd.DataFrame:
# #         """
# #         Identify significant changes in spending habits between two periods.
        
# #         Args:
# #             baseline_start, baseline_end: Baseline period dates (YYYY-MM-DD)
# #             comparison_start, comparison_end: Comparison period dates (YYYY-MM-DD)
# #             threshold: Percentage change considered significant
# #         """
# #         baseline = self.df[self.df['Date'].between(baseline_start, baseline_end)]
# #         comparison = self.df[self.df['Date'].between(comparison_start, comparison_end)]
        
# #         baseline_spending = baseline[baseline['Type'] == 'Expense'].groupby('Main Category')['Amount'].sum().abs()
# #         comparison_spending = comparison[comparison['Type'] == 'Expense'].groupby('Main Category')['Amount'].sum().abs()
        
# #         changes = pd.DataFrame({
# #             'Baseline': baseline_spending,
# #             'Comparison': comparison_spending,
# #             'Change': comparison_spending - baseline_spending,
# #             'Change %': ((comparison_spending - baseline_spending) / baseline_spending) * 100
# #         })
        
# #         return changes[changes['Change %'].abs() > (threshold * 100)].sort_values('Change %', ascending=False)
    
# #     # --------------------------
# #     # Budgeting & Forecasting (61-70)
# #     # --------------------------
    
# #     def project_monthly_spending(self, months: int = 6) -> pd.DataFrame:
# #         """Project future spending based on historical trends."""
# #         monthly = (self.df[self.df['Type'] == 'Expense']
# #                   .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                   .sum()
# #                   .abs())
        
# #         # Simple linear projection
# #         x = np.arange(len(monthly))
# #         y = monthly.values
# #         slope, intercept = np.polyfit(x, y, 1)
        
# #         last_date = monthly.index[-1].to_timestamp()
# #         future_dates = pd.period_range(
# #             start=last_date + pd.DateOffset(months=1),
# #             periods=months,
# #             freq='M'
# #         )
        
# #         future_values = slope * (x[-1] + 1 + np.arange(months)) + intercept
        
# #         return pd.DataFrame({
# #             'Month': future_dates,
# #             'Projected Spending': future_values
# #         }).set_index('Month')
    
# #     def calculate_run_rate(self) -> Dict[str, float]:
# #         """Calculate annual run rate based on current spending patterns."""
# #         monthly_avg = (self.df[self.df['Type'] == 'Expense']
# #                       .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                       .sum()
# #                       .abs()
# #                       .mean())
        
# #         return {
# #             'Monthly Average': monthly_avg,
# #             'Annual Run Rate': monthly_avg * 12
# #         }
    
# #     def budget_vs_actual(self, budget: Dict[str, float]) -> pd.DataFrame:
# #         """
# #         Compare actual spending to budget by category.
        
# #         Args:
# #             budget: Dictionary of {category: budgeted_amount}
# #         """
# #         actual = (self.df[self.df['Type'] == 'Expense']
# #                  .groupby('Main Category')['Amount']
# #                  .sum()
# #                  .abs())
        
# #         comparison = pd.DataFrame({
# #             'Budget': pd.Series(budget),
# #             'Actual': actual,
# #             'Difference': actual - pd.Series(budget),
# #             'Percentage': (actual / pd.Series(budget)) * 100
# #         })
        
# #         return comparison.dropna()
    
# #     def forecast_category_spending(self, category: str, months: int = 6) -> pd.DataFrame:
# #         """Forecast future spending for a specific category."""
# #         category_data = (self.df[(self.df['Type'] == 'Expense') & 
# #                                (self.df['Main Category'] == category)]
# #                         .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                         .sum()
# #                         .abs())
        
# #         if len(category_data) < 2:
# #             return pd.DataFrame()  # Not enough data
        
# #         # Simple linear projection
# #         x = np.arange(len(category_data))
# #         y = category_data.values
# #         slope, intercept = np.polyfit(x, y, 1)
        
# #         last_date = category_data.index[-1].to_timestamp()
# #         future_dates = pd.period_range(
# #             start=last_date + pd.DateOffset(months=1),
# #             periods=months,
# #             freq='M'
# #         )
        
# #         future_values = slope * (x[-1] + 1 + np.arange(months)) + intercept
        
# #         return pd.DataFrame({
# #             'Month': future_dates,
# #             'Projected Spending': future_values
# #         }).set_index('Month')
    
# #     def calculate_savings_potential(self, reduction_targets: Dict[str, float]) -> float:
# #         """
# #         Calculate potential savings based on reduction targets by category.
        
# #         Args:
# #             reduction_targets: Dictionary of {category: percentage_reduction}
# #         """
# #         current_spending = (self.df[self.df['Type'] == 'Expense']
# #                           .groupby('Main Category')['Amount']
# #                           .sum()
# #                           .abs())
        
# #         savings = 0
# #         for category, reduction in reduction_targets.items():
# #             if category in current_spending:
# #                 savings += current_spending[category] * (reduction / 100)
        
# #         return savings
    
# #     def spending_velocity(self, window: str = '30D') -> pd.DataFrame:
# #         """
# #         Calculate spending velocity (amount spent per time window).
        
# #         Args:
# #             window: Time window to calculate velocity (e.g., '30D', '7D')
# #         """
# #         daily = (self.df[self.df['Type'] == 'Expense']
# #                 .groupby('Date')['Amount']
# #                 .sum()
# #                 .abs()
# #                 .resample(window)
# #                 .sum())
        
# #         return daily.to_frame('Spending')
    
# #     def estimate_yearly_totals(self) -> Dict[str, float]:
# #         """Estimate yearly totals based on current data."""
# #         monthly_income = (self.df[self.df['Type'] == 'Income']
# #                          .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                          .sum()
# #                          .mean())
        
# #         monthly_expenses = (self.df[self.df['Type'] == 'Expense']
# #                            .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                            .sum()
# #                            .abs()
# #                            .mean())
        
# #         return {
# #             'Estimated Yearly Income': monthly_income * 12,
# #             'Estimated Yearly Expenses': monthly_expenses * 12,
# #             'Estimated Yearly Savings': (monthly_income - monthly_expenses) * 12
# #         }
    
# #     def identify_budget_optimization(self, priority_categories: List[str]) -> pd.DataFrame:
# #         """
# #         Identify potential budget optimization opportunities.
        
# #         Args:
# #             priority_categories: Categories to prioritize (should not be reduced)
# #         """
# #         spending = (self.df[self.df['Type'] == 'Expense']
# #                    .groupby('Main Category')['Amount']
# #                    .sum()
# #                    .abs()
# #                    .sort_values(ascending=False))
        
# #         optimization = pd.DataFrame({
# #             'Current Spending': spending,
# #             'Priority': spending.index.isin(priority_categories),
# #             'Suggested Reduction %': np.where(
# #                 spending.index.isin(priority_categories),
# #                 0,  # No reduction for priority categories
# #                 np.clip((spending / spending.sum()) * 100, 5, 30)  # Suggested reduction %
# #             )
# #         })
        
# #         optimization['Potential Savings'] = (optimization['Current Spending'] * 
# #                                            optimization['Suggested Reduction %'] / 100)
        
# #         return optimization
    
# #     def calculate_financial_ratios(self) -> Dict[str, float]:
# #         """Calculate key financial ratios."""
# #         total_income = self.total_income()
# #         total_expenses = self.total_expenses()
# #         net_savings = total_income - total_expenses
        
# #         return {
# #             'Savings Rate': (net_savings / total_income) * 100 if total_income > 0 else 0,
# #             'Expense Ratio': (total_expenses / total_income) * 100 if total_income > 0 else 0,
# #             'Discretionary Ratio': (self._calculate_discretionary_spending() / total_income) * 100 if total_income > 0 else 0
# #         }
    
# #     def _calculate_discretionary_spending(self) -> float:
# #         """Calculate discretionary spending (non-essential categories)."""
# #         discretionary_categories = ['Entertainment', 'Shopping & Retail', 'Food & Dining']
# #         return (self.df[(self.df['Type'] == 'Expense') & 
# #                        (self.df['Main Category'].isin(discretionary_categories))]['Amount']
# #                 .sum()
# #                 .abs())
    
# #     # --------------------------
# #     # Advanced Analytics (71-80)
# #     # --------------------------
    
# #     def spending_clusters(self, n_clusters: int = 4) -> pd.DataFrame:
# #         """
# #         Identify spending clusters using k-means clustering.
        
# #         Args:
# #             n_clusters: Number of clusters to identify
# #         """
# #         from sklearn.cluster import KMeans
        
# #         # Prepare data (monthly spending by category)
# #         monthly_cats = (self.df[self.df['Type'] == 'Expense']
# #                        .groupby([self.df['Date'].dt.to_period('M'), 'Main Category'])['Amount']
# #                        .sum()
# #                        .unstack()
# #                        .abs()
# #                        .fillna(0))
        
# #         if len(monthly_cats) < n_clusters:
# #             return pd.DataFrame()  # Not enough data
        
# #         # Normalize data
# #         normalized = (monthly_cats - monthly_cats.mean()) / monthly_cats.std()
        
# #         # Perform clustering
# #         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# #         clusters = kmeans.fit_predict(normalized)
        
# #         # Analyze clusters
# #         monthly_cats['Cluster'] = clusters
# #         cluster_means = monthly_cats.groupby('Cluster').mean()
        
# #         return cluster_means
    
# #     def spending_correlation_matrix(self) -> pd.DataFrame:
# #         """Calculate correlation between spending categories."""
# #         monthly_cats = (self.df[self.df['Type'] == 'Expense']
# #                        .groupby([self.df['Date'].dt.to_period('M'), 'Main Category'])['Amount']
# #                        .sum()
# #                        .unstack()
# #                        .abs()
# #                        .fillna(0))
        
# #         return monthly_cats.corr()
    
# #     def predict_future_spending(self, target_category: str, n_months: int = 3) -> pd.DataFrame:
# #         """
# #         Predict future spending for a category using time series forecasting.
        
# #         Args:
# #             target_category: Category to forecast
# #             n_months: Number of months to predict ahead
# #         """
# #         from statsmodels.tsa.arima.model import ARIMA
        
# #         # Get historical data
# #         history = (self.df[(self.df['Type'] == 'Expense') & 
# #                           (self.df['Main Category'] == target_category)]
# #                   .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                   .sum()
# #                   .abs())
        
# #         if len(history) < 6:  # Need sufficient history
# #             return pd.DataFrame()
        
# #         # Fit ARIMA model (simple configuration)
# #         model = ARIMA(history, order=(1, 0, 1))
# #         model_fit = model.fit()
        
# #         # Make forecast
# #         forecast = model_fit.get_forecast(steps=n_months)
        
# #         # Prepare results
# #         last_date = history.index[-1].to_timestamp()
# #         future_dates = pd.period_range(
# #             start=last_date + pd.DateOffset(months=1),
# #             periods=n_months,
# #             freq='M'
# #         )
        
# #         return pd.DataFrame({
# #             'Month': future_dates,
# #             'Forecast': forecast.predicted_mean,
# #             'Lower CI': forecast.conf_int()['lower Amount'],
# #             'Upper CI': forecast.conf_int()['upper Amount']
# #         }).set_index('Month')
    
# #     def spending_predictors(self, target_category: str) -> pd.DataFrame:
# #         """
# #         Identify which categories best predict spending in a target category.
        
# #         Args:
# #             target_category: Category to predict
# #         """
# #         from sklearn.linear_model import LinearRegression
# #         from sklearn.metrics import r2_score
        
# #         # Prepare monthly data for all categories
# #         monthly_cats = (self.df[self.df['Type'] == 'Expense']
# #                        .groupby([self.df['Date'].dt.to_period('M'), 'Main Category'])['Amount']
# #                        .sum()
# #                        .unstack()
# #                        .abs()
# #                        .fillna(0))
        
# #         if target_category not in monthly_cats.columns:
# #             return pd.DataFrame()
        
# #         # For each other category, see how well it predicts the target
# #         results = []
# #         for predictor in monthly_cats.columns:
# #             if predictor == target_category:
# #                 continue
                
# #             X = monthly_cats[predictor].values.reshape(-1, 1)
# #             y = monthly_cats[target_category].values
            
# #             model = LinearRegression()
# #             model.fit(X, y)
# #             score = r2_score(y, model.predict(X))
            
# #             results.append({
# #                 'Predictor': predictor,
# #                 'R-squared': score,
# #                 'Coefficient': model.coef_[0]
# #             })
        
# #         return pd.DataFrame(results).sort_values('R-squared', ascending=False)
    
# #     def spending_seasonality_decomposition(self, category: str) -> Dict[str, pd.Series]:
# #         """
# #         Decompose spending into trend, seasonal, and residual components.
        
# #         Args:
# #             category: Category to analyze
# #         """
# #         from statsmodels.tsa.seasonal import seasonal_decompose
        
# #         # Get monthly data
# #         monthly = (self.df[(self.df['Type'] == 'Expense') & 
# #                           (self.df['Main Category'] == category)]
# #                   .groupby(self.df['Date'].dt.to_period('M'))['Amount']
# #                   .sum()
# #                   .abs())
        
# #         if len(monthly) < 24:  # Need at least 2 years of data
# #             return {}
        
# #         # Convert to time series
# #         ts = monthly.to_timestamp()
        
# #         # Perform decomposition
# #         decomposition = seasonal_decompose(ts, model='additive', period=12)
        
# #         return {
# #             'observed': decomposition.observed,
# #             'trend': decomposition.trend,
# #             'seasonal': decomposition.seasonal,
# #             'residual': decomposition.resid
# #         }
    
# #     def customer_lifetime_value(self, company: str, discount_rate: float = 0.1) -> float:
# #         """
# #         Estimate customer lifetime value for a company.
        
# #         Args:
# #             company: Company to analyze
# #             discount_rate: Annual discount rate for future cash flows
# #         """
# #         # Get all transactions with this company
# #         company_trans = self.df[self.df['Company'] == company]
        
# #         if company_trans.empty:
# #             return 0.0
        
# #         # Calculate average monthly spending (absolute value)
# #         monthly_spending = (company_trans[company_trans['Type'] == 'Expense']
# #                           .groupby(company_trans['Date'].dt.to_period('M'))['Amount']
# #                           .sum()
# #                           .abs()
# #                           .mean())
        
# #         # Estimate customer lifespan (time between first and last transaction)
# #         lifespan = (company_trans['Date'].max() - company_trans['Date'].min()).days / 365  # in years
        
# #         if lifespan == 0:
# #             lifespan = 1  # At least 1 year if all transactions in same period
        
# #         # Simple CLV calculation (average monthly * 12 * lifespan, discounted)
# #         clv = monthly_spending * 12 * lifespan / ((1 + discount_rate) ** lifespan)
        
# #         return clv
    
# #     def spending_elasticity(self, category1: str, category2: str) -> float:
# #         """
# #         Calculate spending elasticity between two categories.
        
# #         Args:
# #             category1: First category
# #             category2: Second category
# #         """
# #         # Get monthly spending for both categories
# #         monthly = (self.df[self.df['Type'] == 'Expense']
# #                   .groupby([self.df['Date'].dt.to_period('M'), 'Main Category'])['Amount']
# #                   .sum()
# #                   .abs()
# #                   .unstack()
# #                   .fillna(0))
        
# #         if category1 not in monthly.columns or category2 not in monthly.columns:
# #             return np.nan
        
# #         # Calculate percentage changes
# #         pct_change = monthly.pct_change().dropna()
        
# #         if len(pct_change) < 3:  # Need enough data points
# #             return np.nan
        
# #         # Calculate elasticity (beta in regression of category2 ~ category1)
# #         X = pct_change[category1].values.reshape(-1, 1)
# #         y = pct_change[category2].values
        
# #         model = LinearRegression()
# #         model.fit(X, y)
        
# #         return model.coef_[0]
    
# #     def spending_market_basket(self, min_support: float = 0.1) -> pd.DataFrame:
# #         """
# #         Identify frequently co-occurring spending categories using market basket analysis.
        
# #         Args:
# #             min_support: Minimum support threshold for frequent itemsets
# #         """
# #         from mlxtend.frequent_patterns import apriori
# #         from mlxtend.frequent_patterns import association_rules
        
# #         # Create a matrix of categories per transaction
# #         transactions = self.df.copy()
# #         transactions['Category'] = transactions['Main Category']
        
# #         # Create binary matrix (one-hot encoding)
# #         basket = (transactions.groupby(['Date', 'Category'])['Amount']
# #                  .sum()
# #                  .unstack()
# #                  .reset_index()
# #                  .fillna(0)
# #                  .set_index('Date'))
        
# #         # Convert to binary (1 if any spending in category that day)
# #         basket = (basket > 0).astype(int)
        
# #         # Find frequent itemsets
# #         frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
        
# #         if len(frequent_itemsets) == 0:
# #             return pd.DataFrame()
        
# #         # Generate association rules
# #         rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        
# #         return rules.sort_values('lift', ascending=False)
    
# #     def spending_velocity_by_category(self, window: str = '30D') -> pd.DataFrame:
# #         """
# #         Calculate spending velocity by category.
        
# #         Args:
# #             window: Time window to calculate velocity (e.g., '30D', '7D')
# #         """
# #         return (self.df[self.df['Type'] == 'Expense']
# #                 .groupby(['Main Category', pd.Grouper(key='Date', freq=window)])['Amount']
# #                 .sum()
# #                 .abs()
# #                 .groupby('Main Category')
# #                 .mean())
    
# #     # --------------------------
# #     # Visualization Methods
# #     # --------------------------
    
# #     def plot_monthly_trends(self, category: str = None):
# #         """Plot monthly income/expense trends."""
# #         data = self.monthly_trends(category)
# #         if category:
# #             title = f"Monthly {'Spending' if data.mean() < 0 else 'Income'} Trend for {category}"
# #         else:
# #             title = "Monthly Net Cash Flow Trend"
        
# #         data.plot(title=title, figsize=(10, 6))
# #         plt.ylabel('Amount')
# #         plt.grid(True)
# #         plt.show()
    
# #     def plot_category_distribution(self):
# #         """Plot spending distribution by category."""
# #         data = self.category_spending_distribution()
# #         data.sort_values().plot(kind='barh', title='Spending Distribution by Category', figsize=(10, 6))
# #         plt.xlabel('Percentage of Total Spending')
# #         plt.grid(True)
# #         plt.show()
    
# #     def plot_yearly_comparison(self):
# #         """Plot yearly income vs expenses."""
# #         data = self.yearly_summary()
# #         if len(data) < 2:
# #             print("Not enough years to compare")
# #             return
        
# #         data[['Net Balance']].plot(kind='bar', title='Yearly Net Balance', figsize=(10, 6))
# #         plt.ylabel('Amount')
# #         plt.grid(True)
# #         plt.show()
    
# #     def plot_weekday_patterns(self):
# #         """Plot spending patterns by weekday."""
# #         data = self.weekday_analysis()
# #         data.plot(kind='bar', title='Spending Patterns by Weekday', figsize=(10, 6))
# #         plt.ylabel('Amount')
# #         plt.grid(True)
# #         plt.show()
    
# #     def plot_subscription_trends(self):
# #         """Plot subscription spending trends."""
# #         data = self.monthly_subscription_costs()
# #         data.plot(title='Monthly Subscription Costs', figsize=(10, 6))
# #         plt.ylabel('Amount')
# #         plt.grid(True)
# #         plt.show()


# # # Example Usage
# # if __name__ == "__main__":
# #     # Load your data
# #     data = pd.read_csv(r'synthetic_transaction_data.csv', delimiter=',')
    
# #     # For demonstration, we'll create a small sample DataFrame
# #     sample_data = {
# #         'Date': ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15', '2023-03-01'],
# #         'Description': ['Salary', 'Rent', 'Salary', 'Groceries', 'Salary'],
# #         'Amount': [3000, -1000, 3000, -200, 3000],
# #         'Main Category': ['Deposit & Addition', 'Bills & Utilities', 'Deposit & Addition', 'Shopping & Retail', 'Deposit & Addition'],
# #         'Subcategory': ['Salary', 'Rent', 'Salary', 'Groceries', 'Salary'],
# #         'Company': ['Company A', 'Landlord', 'Company A', 'Supermarket', 'Company A']
# #     }
# #     df = pd.DataFrame(data)
    
# #     # Initialize analyzer
# #     analyzer = TransactionAnalyzer(df=df)
    
# #     # Run some analyses
# #     print("Total Income:", analyzer.total_income())
# #     print("Total Expenses:", analyzer.total_expenses())
# #     print("Net Balance:", analyzer.net_balance())
    
# #     print("\nMonthly Trends:")
# #     print(analyzer.monthly_income_expense())
    
# #     print("\nTop Spending Categories:")
# #     print(analyzer.top_expense_categories())
    
# #     # # Generate visualizations
# #     # analyzer.plot_monthly_trends()
# #     # analyzer.plot_category_distribution()
    
    
    
    
    
# # hey look this is how we are prepared data is it????? as we are using the pretrained model bitext/mistral-7B-instructor-banking-v2 so i don't think that we are going to pass this data......you told me that we have to convert this data into the query and reponse format and then stored it and then pass it to this model and then train it????? so this is not perfect i guess???????  look we have to train or finetune this model okay and look i don't want this model is train on only one single query prompt okayy so that's why i was thinking to use the 80-100 different realtime human like queries and answer/repsone, and pass to model so that model get more exposure, what say is it right? and more i want each prompt or query is train on complete data  what i mean to say is let's consider we have 10000 records of bank statement data and we take 100 different real time human like queries okay and 1 query maped with 10000 data okay like wies so at last in or train_data file we have 100*10000 queries and response okayyy..... understood  and one more thing in this file we are not storing the data anywhere so according to in which file format we can store this in json???? think about it cause we are able to store this in csv file right so we have store in json format what say?????   i jsut use other tool but it worst it added visulization and all which i don't want right i want the data which we are use for model train right and look this type of data we have in our synthetic_trasaction_data.csv  Date Description Amount Category Main category Sub categor  01/01 company_name: Uber -230 Deposite & Addition Transport Delivery (more such values) (- debited, + credited)  ........ ........ .......... ............ ....... ............   




















































# import pandas as pd
# import numpy as np
# import json
# import random
# from datetime import datetime, timedelta
# from typing import List, Dict, Any, Optional, Union, Callable
# import re

# class BankingDatasetGenerator:
#     def __init__(self, transaction_data_path: str):
#         """Initialize with path to transaction data CSV"""
#         print(f"Loading transaction data from {transaction_data_path}")
#         self.transactions_df = pd.read_csv(transaction_data_path)
#         self._clean_and_prepare_data()
#         self.query_templates = self._generate_query_templates()
        
#     def _clean_and_prepare_data(self):
#         """Clean and prepare the transaction data"""
#         # Handle date formatting
#         try:
#             self.transactions_df['Date'] = pd.to_datetime(self.transactions_df['Date'], format='%d/%m')
#         except:
#             try:
#                 self.transactions_df['Date'] = pd.to_datetime(self.transactions_df['Date'], format='%m/%d')
#             except:
#                 self.transactions_df['Date'] = pd.to_datetime(self.transactions_df['Date'], errors='coerce')
        
#         # Current year if not in data
#         if self.transactions_df['Date'].dt.year.max() < 2020:
#             current_year = datetime.now().year
#             self.transactions_df['Date'] = self.transactions_df['Date'].apply(
#                 lambda x: x.replace(year=current_year) if not pd.isna(x) else x
#             )
        
#         # Extract company names from Description if needed
#         if not any('company_name:' in str(desc) for desc in self.transactions_df['Description'].sample(min(100, len(self.transactions_df)))):
#             print("Warning: No 'company_name:' pattern found in Description. Using descriptions as-is.")
#         else:
#             self.transactions_df['Company'] = self.transactions_df['Description'].str.extract(r'company_name: ([^,]+)')[0]
        
#         # Ensure Amount is numeric
#         self.transactions_df['Amount'] = pd.to_numeric(self.transactions_df['Amount'], errors='coerce')
        
#         # Extract temporal features
#         self.transactions_df['Month'] = self.transactions_df['Date'].dt.month
#         self.transactions_df['MonthName'] = self.transactions_df['Date'].dt.strftime('%B')
#         self.transactions_df['Year'] = self.transactions_df['Date'].dt.year
#         self.transactions_df['Weekday'] = self.transactions_df['Date'].dt.day_name()
        
#         # Create transaction type
#         self.transactions_df['Transaction_Type'] = np.where(self.transactions_df['Amount'] >= 0, 'Credit', 'Debit')
        
#         print(f"Prepared data with {len(self.transactions_df)} transactions")
        
#     def _generate_query_templates(self) -> List[Dict[str, Any]]:
#         """Generate diverse banking query templates"""
#         templates = []
        
#         # 1. Basic Account Information
#         templates.extend([
#             {
#                 "query_type": "balance_inquiry",
#                 "template": "What's my current account balance?",
#                 "response_function": self._get_balance_response
#             },
#             {
#                 "query_type": "balance_inquiry",
#                 "template": "How much money do I have in my account?",
#                 "response_function": self._get_balance_response
#             },
#             {
#                 "query_type": "balance_inquiry",
#                 "template": "Can you tell me my current balance?",
#                 "response_function": self._get_balance_response
#             },
#             {
#                 "query_type": "transaction_count",
#                 "template": "How many transactions did I make last month?",
#                 "response_function": self._get_transaction_count_response
#             },
#             {
#                 "query_type": "transaction_count",
#                 "template": "Tell me the number of transactions in my account for {month}.",
#                 "response_function": self._get_monthly_transaction_count_response
#             }
#         ])
        
#         # 2. Transaction Search/Query
#         templates.extend([
#             {
#                 "query_type": "transaction_search_company",
#                 "template": "Show me all transactions with {company}",
#                 "response_function": self._get_transaction_search_company_response
#             },
#             {
#                 "query_type": "transaction_search_company",
#                 "template": "How much have I spent at {company} in total?",
#                 "response_function": self._get_company_total_spending_response
#             },
#             {
#                 "query_type": "transaction_search_company",
#                 "template": "Find my most recent transaction with {company}",
#                 "response_function": self._get_latest_company_transaction_response
#             },
#             {
#                 "query_type": "transaction_search_amount",
#                 "template": "Show me transactions above ${amount}",
#                 "response_function": self._get_transactions_above_amount_response
#             },
#             {
#                 "query_type": "transaction_search_amount",
#                 "template": "Are there any transactions under ${amount}?",
#                 "response_function": self._get_transactions_below_amount_response
#             },
#             {
#                 "query_type": "transaction_search_date",
#                 "template": "What did I spend money on {date}?",
#                 "response_function": self._get_spending_by_date_response
#             },
#             {
#                 "query_type": "transaction_search_date_range",
#                 "template": "Show my transactions between {start_date} and {end_date}",
#                 "response_function": self._get_transactions_date_range_response
#             }
#         ])
        
#         # 3. Category-Based Analysis
#         templates.extend([
#             {
#                 "query_type": "spending_category",
#                 "template": "How much did I spend on {category} last month?",
#                 "response_function": self._get_category_spending_response
#             },
#             {
#                 "query_type": "spending_category",
#                 "template": "What percentage of my expenses goes to {category}?",
#                 "response_function": self._get_category_percentage_response
#             },
#             {
#                 "query_type": "spending_category",
#                 "template": "Show me my {category} expenses this year",
#                 "response_function": self._get_yearly_category_expenses_response
#             },
#             {
#                 "query_type": "spending_subcategory",
#                 "template": "How much do I spend on {subcategory}?",
#                 "response_function": self._get_subcategory_spending_response
#             },
#             {
#                 "query_type": "top_categories",
#                 "template": "What are my top spending categories?",
#                 "response_function": self._get_top_spending_categories_response
#             },
#             {
#                 "query_type": "top_categories",
#                 "template": "Where do I spend most of my money?",
#                 "response_function": self._get_top_spending_categories_response
#             }
#         ])
        
#         # 4. Temporal Analysis
#         templates.extend([
#             {
#                 "query_type": "monthly_comparison",
#                 "template": "Compare my spending between {month1} and {month2}",
#                 "response_function": self._get_monthly_comparison_response
#             },
#             {
#                 "query_type": "monthly_analysis",
#                 "template": "How were my finances in {month}?",
#                 "response_function": self._get_monthly_analysis_response
#             },
#             {
#                 "query_type": "monthly_analysis",
#                 "template": "Give me a summary of my transactions for {month}",
#                 "response_function": self._get_monthly_analysis_response
#             },
#             {
#                 "query_type": "weekday_spending",
#                 "template": "What day of the week do I spend the most money?",
#                 "response_function": self._get_weekday_spending_response
#             },
#             {
#                 "query_type": "spending_trend",
#                 "template": "How has my spending on {category} changed over time?",
#                 "response_function": self._get_category_trend_response
#             }
#         ])
        
#         # 5. Budget and Financial Health
#         templates.extend([
#             {
#                 "query_type": "income_vs_expense",
#                 "template": "Am I spending more than I earn?",
#                 "response_function": self._get_income_expense_comparison_response
#             },
#             {
#                 "query_type": "income_vs_expense",
#                 "template": "What's my income to expense ratio?",
#                 "response_function": self._get_income_expense_ratio_response
#             },
#             {
#                 "query_type": "savings_rate",
#                 "template": "What percentage of my income am I saving?",
#                 "response_function": self._get_savings_rate_response
#             },
#             {
#                 "query_type": "largest_expense",
#                 "template": "What's my largest expense this {timeframe}?",
#                 "response_function": self._get_largest_expense_response
#             },
#             {
#                 "query_type": "recurring_expenses",
#                 "template": "Show me my recurring monthly expenses",
#                 "response_function": self._get_recurring_expenses_response
#             }
#         ])
        
#         # 6. Pattern Detection
#         templates.extend([
#             {
#                 "query_type": "unusual_transactions",
#                 "template": "Are there any unusual transactions in my account?",
#                 "response_function": self._get_unusual_transactions_response
#             },
#             {
#                 "query_type": "unusual_transactions",
#                 "template": "Show me any suspicious activity in my account",
#                 "response_function": self._get_unusual_transactions_response
#             },
#             {
#                 "query_type": "spending_habits",
#                 "template": "What are my spending habits?",
#                 "response_function": self._get_spending_habits_response
#             },
#             {
#                 "query_type": "spending_habits",
#                 "template": "Am I an impulsive spender?",
#                 "response_function": self._get_impulsive_spending_response
#             }
#         ])
        
#         # 7. Forecasting and Recommendations
#         templates.extend([
#             {
#                 "query_type": "spending_forecast",
#                 "template": "Based on my history, how much will I spend on {category} next month?",
#                 "response_function": self._get_category_forecast_response
#             },
#             {
#                 "query_type": "savings_forecast",
#                 "template": "How much can I save next month based on my spending patterns?",
#                 "response_function": self._get_savings_forecast_response
#             },
#             {
#                 "query_type": "budget_recommendation",
#                 "template": "What should my monthly budget be for {category}?",
#                 "response_function": self._get_budget_recommendation_response
#             },
#             {
#                 "query_type": "saving_tips",
#                 "template": "How can I save more money?",
#                 "response_function": self._get_saving_tips_response
#             },
#             {
#                 "query_type": "saving_tips",
#                 "template": "Where can I cut expenses?",
#                 "response_function": self._get_expense_cutting_response
#             }
#         ])
        
#         # 8. Advanced Financial Metrics
#         templates.extend([
#             {
#                 "query_type": "cash_flow",
#                 "template": "What's my monthly cash flow?",
#                 "response_function": self._get_cash_flow_response
#             },
#             {
#                 "query_type": "discretionary_spending",
#                 "template": "How much discretionary spending do I have?",
#                 "response_function": self._get_discretionary_spending_response
#             },
#             {
#                 "query_type": "liquidity_ratio",
#                 "template": "What's my liquidity ratio?",
#                 "response_function": self._get_liquidity_ratio_response
#             }
#         ])
        
#         # 9. Bill and Payment Analysis
#         templates.extend([
#             {
#                 "query_type": "bill_payments",
#                 "template": "When did I last pay my {bill_type} bill?",
#                 "response_function": self._get_last_bill_payment_response
#             },
#             {
#                 "query_type": "bill_payments",
#                 "template": "How much do I spend on utilities each month?",
#                 "response_function": self._get_utility_spending_response
#             },
#             {
#                 "query_type": "payment_schedule",
#                 "template": "When are my regular bills due?",
#                 "response_function": self._get_bill_schedule_response
#             }
#         ])
        
#         # 10. Conversational/Human-like Queries
#         templates.extend([
#             {
#                 "query_type": "financial_advice",
#                 "template": "I'm spending too much on dining out. What should I do?",
#                 "response_function": self._get_dining_advice_response
#             },
#             {
#                 "query_type": "financial_advice",
#                 "template": "I want to save for a vacation. How should I plan my finances?",
#                 "response_function": self._get_vacation_savings_response
#             },
#             {
#                 "query_type": "financial_advice",
#                 "template": "I'm trying to pay off debt. What expenses should I cut?",
#                 "response_function": self._get_debt_reduction_advice_response
#             },
#             {
#                 "query_type": "account_summary",
#                 "template": "Summarize my financial situation",
#                 "response_function": self._get_financial_summary_response
#             },
#             {
#                 "query_type": "account_summary",
#                 "template": "Give me an overview of my finances",
#                 "response_function": self._get_financial_summary_response
#             }
#         ])
        
#         # 11. More specific temporal queries
#         templates.extend([
#             {
#                 "query_type": "specific_month_category",
#                 "template": "How much did I spend on {category} in {month}?",
#                 "response_function": self._get_specific_month_category_response
#             },
#             {
#                 "query_type": "weekend_spending",
#                 "template": "Do I spend more on weekends or weekdays?",
#                 "response_function": self._get_weekend_vs_weekday_response
#             },
#             {
#                 "query_type": "daily_spending_average",
#                 "template": "What's my average daily spending?",
#                 "response_function": self._get_daily_spending_average_response
#             },
#             {
#                 "query_type": "monthly_spending_average",
#                 "template": "What's my average monthly spending?",
#                 "response_function": self._get_monthly_spending_average_response
#             }
#         ])
        
#         # 12. Income-specific queries
#         templates.extend([
#             {
#                 "query_type": "income_sources",
#                 "template": "What are my sources of income?",
#                 "response_function": self._get_income_sources_response
#             },
#             {
#                 "query_type": "income_stability",
#                 "template": "Is my income stable month to month?",
#                 "response_function": self._get_income_stability_response
#             },
#             {
#                 "query_type": "income_growth",
#                 "template": "Has my income increased over time?",
#                 "response_function": self._get_income_growth_response
#             }
#         ])
        
#         # 13. Specific comparison queries
#         templates.extend([
#             {
#                 "query_type": "company_comparison",
#                 "template": "Do I spend more at {company1} or {company2}?",
#                 "response_function": self._get_company_comparison_response
#             },
#             {
#                 "query_type": "month_on_month",
#                 "template": "How has my spending changed from last month?",
#                 "response_function": self._get_month_on_month_change_response
#             },
#             {
#                 "query_type": "year_on_year",
#                 "template": "How does my spending this year compare to last year?",
#                 "response_function": self._get_year_on_year_comparison_response
#             }
#         ])
        
#         # 14. Very specific transaction queries
#         templates.extend([
#             {
#                 "query_type": "largest_transaction",
#                 "template": "What was my largest transaction this {timeframe}?",
#                 "response_function": self._get_largest_transaction_response
#             },
#             {
#                 "query_type": "smallest_transaction",
#                 "template": "What was my smallest transaction this {timeframe}?",
#                 "response_function": self._get_smallest_transaction_response
#             },
#             {
#                 "query_type": "average_transaction",
#                 "template": "What's my average transaction amount?",
#                 "response_function": self._get_average_transaction_response
#             },
#             {
#                 "query_type": "transaction_frequency",
#                 "template": "How often do I make transactions?",
#                 "response_function": self._get_transaction_frequency_response
#             }
#         ])
        
#         # 15. Additional human-like queries
#         templates.extend([
#             {
#                 "query_type": "casual_balance",
#                 "template": "Hey, how much money do I have left?",
#                 "response_function": self._get_balance_response
#             },
#             {
#                 "query_type": "casual_spending",
#                 "template": "Did I go overboard with spending last weekend?",
#                 "response_function": self._get_weekend_overspending_response
#             },
#             {
#                 "query_type": "casual_savings",
#                 "template": "Am I saving enough money?",
#                 "response_function": self._get_savings_adequacy_response
#             },
#             {
#                 "query_type": "casual_company",
#                 "template": "I think I shop too much at {company}, right?",
#                 "response_function": self._get_company_frequency_analysis_response
#             }
#         ])
        
#         # 16. Investment-related queries
#         templates.extend([
#             {
#                 "query_type": "investment_tracking",
#                 "template": "How much have I invested this year?",
#                 "response_function": self._get_investment_total_response
#             },
#             {
#                 "query_type": "investment_tracking",
#                 "template": "Show me my investment transactions",
#                 "response_function": self._get_investment_transactions_response
#             }
#         ])
        
#         print(f"Generated {len(templates)} query templates")
#         return templates
    
#     # -------------------------
#     # Response Generation Functions
#     # -------------------------
    
#     # 1. Basic Account Information Responses
    
#     def _get_balance_response(self, data, params=None):
#         """Generate response for balance inquiry"""
#         balance = data["Amount"].sum()
#         if balance >= 0:
#             return f"Your current account balance is ${balance:.2f}."
#         else:
#             return f"Your current account balance is -${abs(balance):.2f}. Your account is overdrawn."
    
#     def _get_transaction_count_response(self, data, params=None):
#         """Generate response for transaction count inquiry"""
#         last_month = data["Date"].max() - pd.Timedelta(days=30)
#         transactions = data[data["Date"] >= last_month]
#         count = len(transactions)
#         return f"You made {count} transactions in the last month."
    
#     def _get_monthly_transaction_count_response(self, data, params):
#         """Generate response for transaction count for a specific month"""
#         month = params["month"]
#         month_num = pd.to_datetime(month, format='%B').month
#         transactions = data[data["Month"] == month_num]
#         count = len(transactions)
#         return f"You made {count} transactions in {month}."
    
#     # 2. Transaction Search/Query Responses
    
#     def _get_transaction_search_company_response(self, data, params):
#         """Generate response for transaction search by company"""
#         company = params["company"]
#         if 'Company' in data.columns:
#             filtered = data[data["Company"].str.contains(company, case=False, na=False)]
#         else:
#             filtered = data[data["Description"].str.contains(company, case=False, na=False)]
            
#         if len(filtered) == 0:
#             return f"No transactions found with {company}."
        
#         total = filtered["Amount"].sum()
#         count = len(filtered)
        
#         response = f"Found {count} transactions with {company}. "
#         if total < 0:
#             response += f"You've spent a total of ${abs(total):.2f} at {company}."
#         else:
#             response += f"You've received a total of ${total:.2f} from {company}."
            
#         return response
    
#     def _get_company_total_spending_response(self, data, params):
#         """Generate response for total spending with a company"""
#         company = params["company"]
#         if 'Company' in data.columns:
#             filtered = data[data["Company"].str.contains(company, case=False, na=False)]
#         else:
#             filtered = data[data["Description"].str.contains(company, case=False, na=False)]
            
#         expenses = filtered[filtered["Amount"] < 0]
#         total_spent = abs(expenses["Amount"].sum())
        
#         if len(expenses) == 0:
#             return f"You haven't spent any money at {company}."
        
#         avg_transaction = total_spent / len(expenses)
#         return f"You've spent a total of ${total_spent:.2f} at {company} across {len(expenses)} transactions. Your average transaction is ${avg_transaction:.2f}."
    
#     def _get_latest_company_transaction_response(self, data, params):
#         """Generate response for latest transaction with a company"""
#         company = params["company"]
#         if 'Company' in data.columns:
#             filtered = data[data["Company"].str.contains(company, case=False, na=False)]
#         else:
#             filtered = data[data["Description"].str.contains(company, case=False, na=False)]
            
#         if len(filtered) == 0:
#             return f"No transactions found with {company}."
        
#         latest = filtered.loc[filtered["Date"].idxmax()]
#         date_str = latest["Date"].strftime("%B %d, %Y")
#         amount = latest["Amount"]
        
#         if amount < 0:
#             return f"Your most recent transaction with {company} was on {date_str} for ${abs(amount):.2f}."
#         else:
#             return f"Your most recent transaction with {company} was on {date_str} for +${amount:.2f}."
    
#     def _get_transactions_above_amount_response(self, data, params):
#         """Generate response for transactions above a certain amount"""
#         amount = float(params["amount"])
#         above_amount = data[data["Amount"].abs() > amount]
        
#         if len(above_amount) == 0:
#             return f"You don't have any transactions above ${amount:.2f}."
        
#         count = len(above_amount)
#         max_amount = above_amount["Amount"].abs().max()
        
#         response = f"You have {count} transactions above ${amount:.2f}. "
#         response += f"Your largest transaction was ${max_amount:.2f}."
        
#         return response
    
#     def _get_transactions_below_amount_response(self, data, params):
#         """Generate response for transactions below a certain amount"""
#         amount = float(params["amount"])
#         below_amount = data[data["Amount"].abs() < amount]
        
#         if len(below_amount) == 0:
#             return f"You don't have any transactions below ${amount:.2f}."
        
#         count = len(below_amount)
#         avg_amount = below_amount["Amount"].abs().mean()
        
#         response = f"You have {count} transactions below ${amount:.2f}. "
#         response += f"The average amount of these smaller transactions is ${avg_amount:.2f}."
        
#         return response
    
#     def _get_spending_by_date_response(self, data, params):
#         """Generate response for spending on a specific date"""
#         date_str = params["date"]
#         try:
#             query_date = pd.to_datetime(date_str)
#             on_date = data[data["Date"].dt.date == query_date.date()]
#         except:
#             # Handle relative dates like "yesterday", "last Monday"
#             today = data["Date"].max()
#             if "yesterday" in date_str.lower():
#                 query_date = today - pd.Timedelta(days=1)
#             elif "last week" in date_str.lower():
#                 query_date = today - pd.Timedelta(days=7)
#             else:
#                 return f"I couldn't understand the date: {date_str}. Please use a format like MM/DD/YYYY."
                
#             on_date = data[data["Date"].dt.date == query_date.date()]
        
#         if len(on_date) == 0:
#             return f"You didn't have any transactions on {query_date.strftime('%B %d, %Y')}."
        
#         expenses = on_date[on_date["Amount"] < 0]
#         total_spent = abs(expenses["Amount"].sum())
        
#         response = f"On {query_date.strftime('%B %d, %Y')}, you made {len(on_date)} transactions. "
        
#         if len(expenses) > 0:
#             response += f"You spent ${total_spent:.2f} on "
#             if 'Main category' in expenses.columns:
#                 categories = expenses['Main category'].unique()
#                 response += ", ".join(categories)
#             else:
#                 companies = expenses['Company'].unique() if 'Company' in expenses.columns else []
#                 if len(companies) > 0:
#                     response += ", ".join(companies)
#                 else:
#                     response += "various purchases"
#         else:
#             response += "You didn't have any expenses on this day."
            
#         return response
    
#     def _get_transactions_date_range_response(self, data, params):
#         """Generate response for transactions in a date range"""
#         start_date = pd.to_datetime(params["start_date"])
#         end_date = pd.to_datetime(params["end_date"])
        
#         in_range = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]
        
#         if len(in_range) == 0:
#             return f"You don't have any transactions between {start_date.strftime('%B %d, %Y')} and {end_date.strftime('%B %d, %Y')}."
        
#         expenses = in_range[in_range["Amount"] < 0]
#         income = in_range[in_range["Amount"] > 0]
        
#         total_spent = abs(expenses["Amount"].sum())
#         total_income = income["Amount"].sum()
        
#         response = f"Between {start_date.strftime('%B %d, %Y')} and {end_date.strftime('%B %d, %Y')}, you had {len(in_range)} transactions. "
#         response += f"You spent ${total_spent:.2f} and received ${total_income:.2f}, "
        
#         net = total_income - total_spent
#         if net >= 0:
#             response += f"for a net gain of ${net:.2f}."
#         else:
#             response += f"for a net loss of ${abs(net):.2f}."
            
#         return response
    
#     # 3. Category-Based Analysis Responses
    
#     def _get_category_spending_response(self, data, params):
#         """Generate spending by category response"""
#         category = params["category"]
#         last_month = data["Date"].max() - pd.Timedelta(days=30)
#         month_data = data[data["Date"] >= last_month]
        
#         if 'Main category' not in data.columns:
#             return f"I don't have category information to answer your question about {category} spending."
        
#         category_data = month_data[month_data["Main category"] == category]
#         category_spend = abs(category_data[category_data["Amount"] < 0]["Amount"].sum())
        
#         if len(category_data) == 0:
#             return f"You haven't spent any money on {category} in the last month."
        
#         total_expenses = abs(month_data[month_data["Amount"] < 0]["Amount"].sum())
#         percentage = (category_spend / total_expenses) * 100 if total_expenses > 0 else 0
        
#         return f"You spent ${category_spend:.2f} on {category} in the last month, which is {percentage:.1f}% of your total expenses."
    
#     def _get_category_percentage_response(self, data, params):
#         """Generate response for category spending percentage"""
#         category = params["category"]
        
#         if 'Main category' not in data.columns:
#             return f"I don't have category information to calculate the percentage spent on {category}."
        
#         expenses = data[data["Amount"] < 0]
#         total_expenses = abs(expenses["Amount"].sum())
        
#         category_expenses = expenses[expenses["Main category"] == category]
#         category_spend = abs(category_expenses["Amount"].sum())
        
#         if total_expenses == 0:
#             return f"You don't have any expenses recorded, so I can't calculate what percentage goes to {category}."
        
#         percentage = (category_spend / total_expenses) * 100
        
#         response = f"{category} makes up {percentage:.1f}% of your total expenses. "
        
#         if percentage > 30:
#             response += f"This is a significant portion of your spending."
#         elif percentage > 15:
#             response += f"This is a moderate portion of your spending."
#         else:
#             response += f"This is a relatively small portion of your spending."
            
#         return response
    
#     def _get_yearly_category_expenses_response(self, data, params):
#         """Generate response for yearly category expenses"""
#         category = params["category"]
        
#         if 'Main category' not in data.columns:
#             return f"I don't have category information to show your {category} expenses."
        
#         current_year = data["Date"].max().year
#         year_data = data[data["Year"] == current_year]
        
#         category_expenses = year_data[(year_data["Main category"] == category) & (year_data["Amount"] < 0)]
#         category_spend = abs(category_expenses["Amount"].sum())
        
#         if len(category_expenses) == 0:
#             return f"You haven't spent any money on {category} this year."
        
#         months_with_expenses = category_expenses["Month"].nunique()
#         avg_monthly = category_spend / months_with_expenses if months_with_expenses > 0 else 0
        
#         response = f"This year, you've spent ${category_spend:.2f} on {category} across {len(category_expenses)} transactions. "
        
#         if months_with_expenses > 1:
#             response += f"Your average monthly {category} expense is ${avg_monthly:.2f}."
        
#         return response
    
#     def _get_subcategory_spending_response(self, data, params):
#         """Generate response for subcategory spending"""
#         subcategory = params["subcategory"]
        
#         if 'Sub category' not in data.columns:
#             return f"I don't have subcategory information to analyze your {subcategory} spending."
        
#         subcategory_expenses = data[(data["Sub category"] == subcategory) & (data["Amount"] < 0)]
#         subcategory_spend = abs(subcategory_expenses["Amount"].sum())
        
#         if len(subcategory_expenses) == 0:
#             return f"You haven't spent any money on {subcategory}."
        
#         avg_transaction = subcategory_spend / len(subcategory_expenses)
        
#         response = f"You've spent a total of ${subcategory_spend:.2f} on {subcategory} across {len(subcategory_expenses)} transactions. "
#         response += f"Your average {subcategory} transaction is ${avg_transaction:.2f}."
        
#         return response
    
#     def _get_top_spending_categories_response(self, data, params=None):
#             """Generate response for top spending categories"""
#             if 'Main category' not in data.columns:
#                 return "I don't have category information to determine your top spending categories."
            
#             expenses = data[data["Amount"] < 0]
            
#             if len(expenses) == 0:
#                 return "You don't have any expenses recorded to determine top spending categories."
            
#             category_spending = expenses.groupby('Main category')['Amount'].sum().abs().sort_values(ascending=False)
#             top_categories = category_spending.head(5)
#             total_expenses = abs(expenses["Amount"].sum())
            
#             response = "Your top spending categories are:\n"
#             for category, amount in top_categories.items():
#                 percentage = (amount / total_expenses) * 100
#                 response += f"- {category}: ${amount:.2f} ({percentage:.1f}% of total spending)\n"
            
#             return response
    
#     # 4. Temporal Analysis Responses
    
#     def _get_monthly_comparison_response(self, data, params):
#         """Generate response comparing spending between two months"""
#         month1 = params["month1"]
#         month2 = params["month2"]
        
#         month1_num = pd.to_datetime(month1, format='%B').month
#         month2_num = pd.to_datetime(month2, format='%B').month
        
#         month1_data = data[data["Month"] == month1_num]
#         month2_data = data[data["Month"] == month2_num]
        
#         month1_expenses = abs(month1_data[month1_data["Amount"] < 0]["Amount"].sum())
#         month2_expenses = abs(month2_data[month2_data["Amount"] < 0]["Amount"].sum())
        
#         difference = abs(month1_expenses - month2_expenses)
#         percent_change = (difference / month1_expenses) * 100 if month1_expenses > 0 else 0
        
#         response = f"In {month1}, you spent ${month1_expenses:.2f}. In {month2}, you spent ${month2_expenses:.2f}. "
        
#         if month1_expenses > month2_expenses:
#             response += f"You spent ${difference:.2f} ({percent_change:.1f}%) less in {month2} compared to {month1}."
#         elif month2_expenses > month1_expenses:
#             response += f"You spent ${difference:.2f} ({percent_change:.1f}%) more in {month2} compared to {month1}."
#         else:
#             response += f"Your spending was exactly the same in both months."
            
#         return response
    
#     def _get_monthly_analysis_response(self, data, params):
#         """Generate response for monthly financial analysis"""
#         month = params["month"]
#         month_num = pd.to_datetime(month, format='%B').month
        
#         month_data = data[data["Month"] == month_num]
        
#         if len(month_data) == 0:
#             return f"You don't have any transactions recorded for {month}."
        
#         expenses = month_data[month_data["Amount"] < 0]
#         income = month_data[month_data["Amount"] > 0]
        
#         total_expenses = abs(expenses["Amount"].sum())
#         total_income = income["Amount"].sum()
#         net = total_income - total_expenses
        
#         response = f"In {month}, you spent ${total_expenses:.2f} and received ${total_income:.2f}, "
#         if net >= 0:
#             response += f"for a net gain of ${net:.2f}. "
#         else:
#             response += f"for a net loss of ${abs(net):.2f}. "
        
#         if 'Main category' in month_data.columns:
#             top_category = expenses.groupby('Main category')['Amount'].sum().abs().idxmax() if len(expenses) > 0 else None
            
#             if top_category:
#                 top_category_amount = abs(expenses[expenses['Main category'] == top_category]['Amount'].sum())
#                 response += f"Your highest spending category was {top_category} at ${top_category_amount:.2f}."
        
#         return response
    
#     def _get_weekday_spending_response(self, data, params=None):
#         """Generate response for weekday spending analysis"""
#         expenses = data[data["Amount"] < 0]
        
#         if len(expenses) == 0:
#             return "You don't have any expenses recorded to analyze weekday spending patterns."
        
#         weekday_spending = expenses.groupby('Weekday')['Amount'].sum().abs()
        
#         # Ensure proper ordering of days
#         weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#         weekday_spending = weekday_spending.reindex(weekdays)
        
#         highest_day = weekday_spending.idxmax()
#         highest_amount = weekday_spending.max()
        
#         lowest_day = weekday_spending.idxmin()
#         lowest_amount = weekday_spending.min()
        
#         response = f"You spend the most on {highest_day}s, with an average of ${highest_amount:.2f}. "
#         response += f"You spend the least on {lowest_day}s, with an average of ${lowest_amount:.2f}."
        
#         return response
    
#     def _get_category_trend_response(self, data, params):
#         """Generate response for category spending trend over time"""
#         category = params["category"]
        
#         if 'Main category' not in data.columns:
#             return f"I don't have category information to analyze trends for {category}."
        
#         category_expenses = data[(data["Main category"] == category) & (data["Amount"] < 0)]
        
#         if len(category_expenses) == 0:
#             return f"You haven't spent any money on {category}."
        
#         # Monthly trend
#         monthly_trend = category_expenses.groupby(['Year', 'Month'])['Amount'].sum().abs()
        
#         if len(monthly_trend) < 2:
#             return f"You don't have enough historical data to analyze trends for {category}."
        
#         first_month_amount = monthly_trend.iloc[0]
#         last_month_amount = monthly_trend.iloc[-1]
        
#         overall_change = last_month_amount - first_month_amount
#         percent_change = (overall_change / first_month_amount) * 100 if first_month_amount > 0 else 0
        
#         response = f"Your spending on {category} "
        
#         if overall_change > 0:
#             response += f"has increased by ${overall_change:.2f} ({percent_change:.1f}%) over time."
#         elif overall_change < 0:
#             response += f"has decreased by ${abs(overall_change):.2f} ({abs(percent_change):.1f}%) over time."
#         else:
#             response += f"has remained stable over time."
        
#         return response
    
#     # 5. Budget and Financial Health Responses
    
#     def _get_income_expense_comparison_response(self, data, params=None):
#         """Generate response comparing income to expenses"""
#         expenses = data[data["Amount"] < 0]
#         income = data[data["Amount"] > 0]
        
#         total_expenses = abs(expenses["Amount"].sum())
#         total_income = income["Amount"].sum()
        
#         if total_income == 0:
#             return "You don't have any income recorded, so I can't compare income to expenses."
        
#         net = total_income - total_expenses
#         expense_to_income_ratio = total_expenses / total_income
        
#         response = f"You've spent ${total_expenses:.2f} and earned ${total_income:.2f}, "
        
#         if net >= 0:
#             response += f"giving you a net gain of ${net:.2f}. "
#             response += f"You're spending {expense_to_income_ratio:.1%} of your income."
#         else:
#             response += f"giving you a net loss of ${abs(net):.2f}. "
#             response += f"You're spending {expense_to_income_ratio:.1%} of your income, which is more than you earn."
        
#         return response
    
#     def _get_income_expense_ratio_response(self, data, params=None):
#         """Generate response for income to expense ratio"""
#         expenses = data[data["Amount"] < 0]
#         income = data[data["Amount"] > 0]
        
#         total_expenses = abs(expenses["Amount"].sum())
#         total_income = income["Amount"].sum()
        
#         if total_income == 0:
#             return "You don't have any income recorded, so I can't calculate an income to expense ratio."
        
#         expense_to_income_ratio = total_expenses / total_income
        
#         response = f"Your expense to income ratio is {expense_to_income_ratio:.2f}. "
        
#         if expense_to_income_ratio < 0.5:
#             response += "This is excellent! You're spending less than half of what you earn."
#         elif expense_to_income_ratio < 0.7:
#             response += "This is good. You're saving a decent portion of your income."
#         elif expense_to_income_ratio < 1:
#             response += "You're spending most of your income, but still saving some."
#         else:
#             response += "You're spending more than you earn. Consider reviewing your expenses."
        
#         return response
    
#     def _get_savings_rate_response(self, data, params=None):
#         """Generate response for savings rate"""
#         expenses = data[data["Amount"] < 0]
#         income = data[data["Amount"] > 0]
        
#         total_expenses = abs(expenses["Amount"].sum())
#         total_income = income["Amount"].sum()
        
#         if total_income == 0:
#             return "You don't have any income recorded, so I can't calculate your savings rate."
        
#         savings = total_income - total_expenses
#         savings_rate = (savings / total_income) * 100 if total_income > 0 else 0
        
#         response = f"Your savings rate is {savings_rate:.1f}% of your income. "
        
#         if savings_rate >= 20:
#             response += "This is excellent! Financial experts recommend saving at least 20% of your income."
#         elif savings_rate >= 10:
#             response += "This is good. You're on the right track, but could aim to save more if possible."
#         elif savings_rate > 0:
#             response += "You're saving some money, but should try to increase your savings rate to at least 10-20%."
#         else:
#             response += "You're not saving any money. Consider reviewing your expenses to find areas to cut back."
        
#         return response
    
#     def _get_largest_expense_response(self, data, params):
#         """Generate response for largest expense in a time period"""
#         timeframe = params["timeframe"]
        
#         if timeframe.lower() == "month":
#             period_data = data[data["Month"] == data["Date"].max().month]
#         elif timeframe.lower() == "year":
#             period_data = data[data["Year"] == data["Date"].max().year]
#         elif timeframe.lower() == "week":
#             period_data = data[data["Date"] >= (data["Date"].max() - pd.Timedelta(days=7))]
#         else:
#             return f"I couldn't understand the timeframe: {timeframe}. Please use 'month', 'year', or 'week'."
        
#         expenses = period_data[period_data["Amount"] < 0]
        
#         if len(expenses) == 0:
#             return f"You don't have any expenses recorded for this {timeframe}."
        
#         largest_expense = expenses.loc[expenses["Amount"].abs().idxmax()]
#         amount = abs(largest_expense["Amount"])
#         date = largest_expense["Date"].strftime("%B %d, %Y")
        
#         response = f"Your largest expense this {timeframe} was ${amount:.2f} on {date}. "
        
#         if 'Company' in largest_expense.index and not pd.isna(largest_expense["Company"]):
#             response += f"This was a transaction with {largest_expense['Company']}."
#         elif 'Description' in largest_expense.index:
#             response += f"Description: {largest_expense['Description']}"
        
#         return response
    
#     def _get_recurring_expenses_response(self, data, params=None):
#         """Generate response for recurring monthly expenses"""
#         # This is a simplified approach - real implementation would need more sophisticated pattern recognition
#         expenses = data[data["Amount"] < 0].copy()
        
#         if len(expenses) == 0:
#             return "You don't have any expenses recorded to identify recurring payments."
        
#         # Find transactions with similar amounts across multiple months
#         if 'Company' in expenses.columns:
#             grouped = expenses.groupby('Company')
#         else:
#             # Extract likely merchants from description
#             expenses['Merchant'] = expenses['Description'].str.extract(r'^([A-Za-z\s]+)')[0]
#             grouped = expenses.groupby('Merchant')
        
#         recurring = []
        
#         for name, group in grouped:
#             if len(group) >= 2 and group['Month'].nunique() >= 2:
#                 amounts = group['Amount'].abs().values
#                 # Check for consistent amounts (within 5% variance)
#                 if (max(amounts) - min(amounts)) / max(amounts) < 0.05:
#                     avg_amount = group['Amount'].abs().mean()
#                     recurring.append((name, avg_amount, len(group)))
        
#         if not recurring:
#             return "I couldn't identify any clear recurring monthly expenses in your transactions."
        
#         recurring.sort(key=lambda x: x[1], reverse=True)
        
#         response = "Your recurring monthly expenses appear to be:\n"
#         for name, amount, count in recurring[:5]:  # Show top 5
#             response += f"- {name}: ${amount:.2f} (appeared {count} times)\n"
        
#         total = sum(item[1] for item in recurring)
#         response += f"\nThese recurring expenses total approximately ${total:.2f} per month."
        
#         return response
    
#     # 6. Pattern Detection Responses
    
#     def _get_unusual_transactions_response(self, data, params=None):
#         """Generate response for unusual transactions"""
#         # Again, simplified approach - real implementation would need more sophisticated anomaly detection
#         expenses = data[data["Amount"] < 0]
        
#         if len(expenses) < 5:  # Need sufficient data
#             return "You don't have enough transaction history to detect unusual patterns."
        
#         # Find transactions significantly larger than average
#         mean = expenses["Amount"].abs().mean()
#         std = expenses["Amount"].abs().std()
#         threshold = mean + (2 * std)  # Transactions 2+ standard deviations above mean
        
#         unusual = expenses[expenses["Amount"].abs() > threshold]
        
#         if len(unusual) == 0:
#             return "I didn't detect any unusual transactions in your account."
        
#         response = f"I found {len(unusual)} potentially unusual transactions that are significantly larger than your typical spending:\n"
        
#         # Show top 3 most unusual
#         unusual = unusual.sort_values(by="Amount", ascending=True)
#         for _, row in unusual.head(3).iterrows():
#             date = row["Date"].strftime("%B %d, %Y")
#             amount = abs(row["Amount"])
#             if 'Company' in row.index and not pd.isna(row["Company"]):
#                 merchant = row["Company"]
#             else:
#                 merchant = row["Description"]
            
#             response += f"- ${amount:.2f} at {merchant} on {date}\n"
        
#         return response
    
#     def _get_spending_habits_response(self, data, params=None):
#         """Generate response for spending habits analysis"""
#         expenses = data[data["Amount"] < 0]
        
#         if len(expenses) < 10:  # Need sufficient data
#             return "You don't have enough transaction history to analyze spending habits."
        
#         # Average transaction amount
#         avg_transaction = expenses["Amount"].abs().mean()
        
#         # Frequency of transactions
#         date_range = (expenses["Date"].max() - expenses["Date"].min()).days
#         avg_daily_transactions = len(expenses) / (date_range if date_range > 0 else 1)
        
#         # Day of week patterns
#         weekday_counts = expenses["Weekday"].value_counts()
#         most_common_day = weekday_counts.idxmax()
        
#         # Time trends
#         recent = expenses[expenses["Date"] >= (expenses["Date"].max() - pd.Timedelta(days=30))]
#         older = expenses[expenses["Date"] < (expenses["Date"].max() - pd.Timedelta(days=30))]
        
#         recent_avg = recent["Amount"].abs().mean() if len(recent) > 0 else 0
#         older_avg = older["Amount"].abs().mean() if len(older) > 0 else 0
        
#         trend = "increasing" if recent_avg > older_avg * 1.1 else "decreasing" if recent_avg < older_avg * 0.9 else "stable"
        
#         response = "Based on your transaction history, here are your spending habits:\n"
#         response += f"- You make about {avg_daily_transactions:.1f} transactions per day\n"
#         response += f"- Your average transaction is ${avg_transaction:.2f}\n"
#         response += f"- You tend to spend most often on {most_common_day}s\n"
#         response += f"- Your spending appears to be {trend} over time\n"
        
#         if 'Main category' in expenses.columns:
#             top_category = expenses.groupby('Main category')['Amount'].sum().abs().idxmax()
#             response += f"- Your highest spending category is {top_category}"
        
#         return response
    
#     def _get_impulsive_spending_response(self, data, params=None):
#         """Generate response for impulsive spending assessment"""
#         expenses = data[data["Amount"] < 0]
        
#         if len(expenses) < 20:  # Need sufficient data
#             return "You don't have enough transaction history to assess impulsive spending patterns."
        
#         # Small frequent purchases often indicate impulsive spending
#         small_purchases = expenses[expenses["Amount"].abs() < expenses["Amount"].abs().mean() * 0.5]
#         small_purchase_ratio = len(small_purchases) / len(expenses)
        
#         # Weekend/evening spending might indicate impulsive purchases
#         weekend_spending = expenses[expenses["Weekday"].isin(["Saturday", "Sunday"])]
#         weekend_ratio = len(weekend_spending) / len(expenses)
        
#         # Category analysis (if available)
#         if 'Main category' in expenses.columns:
#             impulsive_categories = ["Entertainment", "Dining", "Shopping", "Takeout"]
#             impulsive_spending = expenses[expenses["Main category"].isin(impulsive_categories)]
#             impulsive_ratio = len(impulsive_spending) / len(expenses)
#         else:
#             impulsive_ratio = None
        
#         impulsive_indicators = 0
#         if small_purchase_ratio > 0.4:
#             impulsive_indicators += 1
#         if weekend_ratio > 0.4:
#             impulsive_indicators += 1
#         if impulsive_ratio is not None and impulsive_ratio > 0.3:
#             impulsive_indicators += 1
        
#         response = ""
#         if impulsive_indicators >= 2:
#             response = "Based on your spending patterns, you may have some impulsive spending tendencies. "
#             response += f"{small_purchase_ratio:.0%} of your transactions are smaller purchases, and "
#             response += f"{weekend_ratio:.0%} of your spending happens on weekends. "
            
#             if impulsive_ratio is not None:
#                 response += f"Additionally, {impulsive_ratio:.0%} of your spending is in categories often associated with impulse buying."
#         else:
#             response = "Based on your spending patterns, you don't appear to be an impulsive spender. "
#             response += "Your purchases tend to be planned and consistent."
        
#         return response
    
#     # 7. Forecasting and Recommendations Responses
    
#     def _get_category_forecast_response(self, data, params):
#         """Generate response for category spending forecast"""
#         category = params["category"]
        
#         if 'Main category' not in data.columns:
#             return f"I don't have category information to forecast {category} spending."
        
#         category_expenses = data[(data["Main category"] == category) & (data["Amount"] < 0)]
        
#         if len(category_expenses) < 3:
#             return f"You don't have enough historical data on {category} spending to make a forecast."
        
#         # Simple moving average forecast
#         monthly_spending = category_expenses.groupby(['Year', 'Month'])['Amount'].sum().abs()
        
#         if len(monthly_spending) < 3:
#             return f"You don't have enough monthly data on {category} to make a reliable forecast."
        
#         last_3_months_avg = monthly_spending.tail(3).mean()
#         trend = monthly_spending.pct_change().mean()
        
#         # Adjust forecast based on trend
#         forecast = last_3_months_avg * (1 + trend)
        
#         response = f"Based on your spending history, I forecast you'll spend about ${forecast:.2f} on {category} next month. "
        
#         if trend > 0.05:
#             response += f"This represents an increase of {trend:.1%} based on your recent upward trend in this category."
#         elif trend < -0.05:
#             response += f"This represents a decrease of {abs(trend):.1%} based on your recent downward trend in this category."
#         else:
#             response += "This is consistent with your recent spending patterns in this category."
        
#         return response
    
#     def _get_savings_forecast_response(self, data, params=None):
#         """Generate response for savings forecast"""
#         # Get recent months data for better forecast
#         recent_months = 3
#         recent_data = data[data["Date"] >= (data["Date"].max() - pd.Timedelta(days=30 * recent_months))]
        
#         if len(recent_data) < 10:
#             return "You don't have enough transaction history to make a reliable savings forecast."
        
#         # Calculate monthly income and expenses
#         monthly_income = recent_data[recent_data["Amount"] > 0].groupby(['Year', 'Month'])['Amount'].sum()
#         monthly_expenses = recent_data[recent_data["Amount"] < 0].groupby(['Year', 'Month'])['Amount'].sum().abs()
        
#         if len(monthly_income) < 2 or len(monthly_expenses) < 2:
#             return "You don't have enough monthly data to make a reliable savings forecast."
        
#         avg_monthly_income = monthly_income.mean()
#         avg_monthly_expenses = monthly_expenses.mean()
        
#         expected_savings = avg_monthly_income - avg_monthly_expenses
        
#         response = f"Based on your recent financial activity, you can expect to save about ${expected_savings:.2f} next month. "
        
#         if expected_savings <= 0:
#             response += "Unfortunately, your expenses are currently exceeding your income, making it difficult to save."
#         elif expected_savings < avg_monthly_income * 0.1:
#             response += "You're saving a small amount each month, but could benefit from reducing expenses to increase savings."
#         else:
#             response += f"This represents a savings rate of {(expected_savings / avg_monthly_income) * 100:.1f}% of your income."
        
#         return response
    
#     def _get_budget_recommendation_response(self, data, params):
#         """Generate response for budget recommendation"""
#         category = params["category"]
        
#         if 'Main category' not in data.columns:
#             return f"I don't have category information to recommend a budget for {category}."
        
#         category_expenses = data[(data["Main category"] == category) & (data["Amount"] < 0)]
        
#         if len(category_expenses) < 3:
#             return f"You don't have enough historical data on {category} spending to recommend a budget."
        
#         monthly_spending = category_expenses.groupby(['Year', 'Month'])['Amount'].sum().abs()
        
#         avg_monthly = monthly_spending.mean()
#         min_monthly = monthly_spending.min()
#         max_monthly = monthly_spending.max()
        
#         # Total monthly expenses
#         total_monthly_expenses = data[data["Amount"] < 0].groupby(['Year', 'Month'])['Amount'].sum().abs().mean()
        
#         # Recommended budget based on historical spending but slightly reduced
#         recommended = avg_monthly * 0.9
        
#         # Check if recommendation is reasonable (between minimum and average)
#         if recommended < min_monthly:
#             recommended = min_monthly
        
#         response = f"Based on your spending history, I recommend a monthly budget of ${recommended:.2f} for {category}. "
#         response += f"This is {(recommended / total_monthly_expenses) * 100:.1f}% of your typical monthly expenses. "
#         response += f"For context, you've spent between ${min_monthly:.2f} and ${max_monthly:.2f} per month on {category} in the past."
        
#         return response
    
#     def _get_saving_tips_response(self, data, params=None):
#         """Generate response with saving tips"""
#         expenses = data[data["Amount"] < 0]
        
#         if len(expenses) < 20:  # Need sufficient data
#             return "Here are some general saving tips:\n1. Track all expenses\n2. Create a budget\n3. Reduce unnecessary subscriptions\n4. Plan meals and cook at home\n5. Use cashback and rewards programs"
        
#         tips = ["Here are some personalized saving tips based on your spending patterns:"]
        
#         # Check for category-specific advice
#         if 'Main category' in expenses.columns:
#             category_spending = expenses.groupby('Main category')['Amount'].sum().abs().sort_values(ascending=False)
#             top_categories = category_spending.head(3)
            
#             for category, amount in top_categories.items():
#                 if category == "Dining" or category == "Restaurants":
#                     tips.append(f"Consider reducing your ${amount:.2f} {category} expenses by cooking more meals at home")
#                 elif category == "Entertainment":
#                     tips.append(f"Your ${amount:.2f} {category} spending could be reduced by finding free or less expensive activities")
#                 elif category == "Shopping":
#                     tips.append(f"Consider implementing a 48-hour rule before making non-essential purchases to reduce your ${amount:.2f} {category} expenses")
        
#         # Check for frequent small transactions
#         small_transactions = expenses[expenses["Amount"].abs() < expenses["Amount"].abs().mean() * 0.3]
#         if len(small_transactions) / len(expenses) > 0.3:
#             tips.append("You have many small transactions that can add up - consider consolidating purchases")
        
#         # Check for weekend spending
#         weekend_spending = expenses[expenses["Weekday"].isin(["Saturday", "Sunday"])]
#         if len(weekend_spending) / len(expenses) > 0.4:
#             tips.append("A significant portion of your spending happens on weekends - plan your weekend activities in advance to avoid impulse spending")
        
#         # General tips
#         tips.append("Review your subscriptions and cancel those you don't use regularly")
#         tips.append("Set up automatic transfers to a savings account on payday")
#         tips.append("Use cash for discretionary spending to make yourself more conscious of purchases")
        
#         return "\n".join(tips)
    
#     def _get_expense_cutting_response(self, data, params=None):
#         """Generate response with expense cutting suggestions"""
#         expenses = data[data["Amount"] < 0]
        
#         if len(expenses) < 20:  # Need sufficient data
#             return "Without extensive transaction history, I recommend reviewing subscriptions, dining out less frequently, reducing impulse purchases, and considering cheaper alternatives for your regular expenses."
        
#         suggestions = ["Based on your transaction history, here are areas where you might cut expenses:"]
        
#         # Check for category-specific suggestions
#         if 'Main category' in expenses.columns:
#             category_spending = expenses.groupby('Main category')['Amount'].sum().abs().sort_values(ascending=False)
#             top_categories = category_spending.head(5)
            
#             for category, amount in top_categories.items():
#                 if category == "Dining" or category == "Restaurants":
#                     suggestions.append(f"${amount:.2f} on {category} - Cook more meals at home or bring lunch to work")
#                 elif category == "Entertainment":
#                     suggestions.append(f"${amount:.2f} on {category} - Look for free events or use streaming services instead of multiple subscriptions")
#                 elif category == "Shopping":
#                     suggestions.append(f"${amount:.2f} on {category} - Implement a waiting period before purchases and focus on needs vs. wants")
#                 elif category == "Groceries":
#                     suggestions.append(f"${amount:.2f} on {category} - Plan meals, use a shopping list, and buy in bulk when possible")
#                 elif category == "Transportation":
#                     suggestions.append(f"${amount:.2f} on {category} - Consider carpooling, public transit, or combining errands")
        
#         # Check for recurring small expenses
#         if 'Company' in expenses.columns:
#             frequent_merchants = expenses['Company'].value_counts().head(5)
#             for merchant, count in frequent_merchants.items():
#                 if count >= 5:
#                     merchant_total = expenses[expenses['Company'] == merchant]['Amount'].abs().sum()
#                     suggestions.append(f"You visited {merchant} {count} times, spending ${merchant_total:.2f} - Consider reducing frequency")
        
#         # General suggestions if list is short
#         if len(suggestions) < 3:
#             suggestions.append("Review all subscriptions and cancel those you don't regularly use")
#             suggestions.append("Consider negotiating bills like insurance, internet, and phone plans")
#             suggestions.append("Look for energy-saving opportunities in your home")
        
#         return "\n".join(suggestions)
    
#     # 8. Advanced Financial Metrics Responses
    
#     def _get_cash_flow_response(self, data, params=None):
#         """Generate response for monthly cash flow"""
#         # Calculate monthly income and expenses
#         monthly_data = data.groupby(['Year', 'Month'])
        
#         if len(monthly_data) < 1:
#             return "You don't have enough monthly data to calculate cash flow."
        
#         monthly_cash_flows = []
        
#         for (year, month), month_data in monthly_data:
#             income = month_data[month_data["Amount"] > 0]["Amount"].sum()
#             expenses = abs(month_data[month_data["Amount"] < 0]["Amount"].sum())
#             cash_flow = income - expenses
#             monthly_cash_flows.append(cash_flow)
        
#         average_cash_flow = sum(monthly_cash_flows) / len(monthly_cash_flows)
        
#         response = f"Your average monthly cash flow is ${average_cash_flow:.2f}. "
        
#         if average_cash_flow > 0:
#             response += f"This is positive, meaning you typically have ${average_cash_flow:.2f} left over each month."
#         else:
#             response += f"This is negative, meaning you typically spend ${abs(average_cash_flow):.2f} more than you earn each month."
        
#         return response
    
# def get_discretionary_spending_response(self, data, params=None):
#     """
#     Analyzes discretionary spending data and generates a response with insights.
    
#     Args:
#         data (dict): Dictionary containing spending data by category
#         params (dict, optional): Additional parameters for customizing the analysis
    
#     Returns:
#         dict: A response containing analysis results and insights
#     """
#     if not data:
#         return {"status": "error", "message": "No data provided for analysis"}
    
#     if params is None:
#         params = {}
    
#     # Extract default parameters or use provided ones
#     time_period = params.get("time_period", "monthly")
#     threshold = params.get("threshold", 0.1)  # 10% threshold for significant changes
#     top_n = params.get("top_n", 5)  # Number of top categories to include
    
#     # Initialize response structure
#     response = {
#         "status": "success",
#         "summary": {},
#         "top_categories": [],
#         "insights": [],
#         "recommendations": []
#     }
    
#     # Calculate total discretionary spending
#     total_spending = sum(data.values())
#     response["summary"]["total"] = total_spending
    
#     # Find top spending categories
#     sorted_categories = sorted(data.items(), key=lambda x: x[1], reverse=True)
#     top_categories = sorted_categories[:top_n]
    
#     response["top_categories"] = [
#         {"category": cat, "amount": amount, "percentage": (amount / total_spending * 100)}
#         for cat, amount in top_categories
#     ]
    
#     # Generate insights based on spending patterns
#     if len(data) > 1:
#         # Calculate average spending per category
#         avg_spending = total_spending / len(data)
        
#         # Identify categories with significantly higher spending
#         high_spending = [cat for cat, amount in data.items() 
#                          if amount > avg_spending * (1 + threshold)]
        
#         if high_spending:
#             response["insights"].append({
#                 "type": "high_spending",
#                 "message": f"Higher than average spending in {', '.join(high_spending)}",
#                 "categories": high_spending
#             })
        
#         # Calculate percentage of total for top category
#         top_category, top_amount = sorted_categories[0]
#         top_percentage = top_amount / total_spending * 100
        
#         if top_percentage > 40:  # If top category takes over 40% of discretionary spending
#             response["insights"].append({
#                 "type": "concentration",
#                 "message": f"{top_category} accounts for {top_percentage:.1f}% of your discretionary spending",
#                 "category": top_category,
#                 "percentage": top_percentage
#             })
    
#     # Generate recommendations
#     if total_spending > 0:
#         response["recommendations"] = self._generate_spending_recommendations(data, total_spending)
    
#     return response

# def _generate_spending_recommendations(self, data, total_spending):
#     """
#     Helper method to generate recommendations based on spending patterns.
    
#     Args:
#         data (dict): Dictionary containing spending data by category
#         total_spending (float): Total discretionary spending amount
    
#     Returns:
#         list: List of recommendation objects
#     """
#     recommendations = []
    
#     # Sort categories by spending amount (descending)
#     sorted_categories = sorted(data.items(), key=lambda x: x[1], reverse=True)
    
#     # If the top category takes more than 30% of spending
#     if sorted_categories and sorted_categories[0][1] / total_spending > 0.3:
#         top_category = sorted_categories[0][0]
#         recommendations.append({
#             "type": "reduce",
#             "category": top_category,
#             "message": f"Consider reducing spending in {top_category} to balance your discretionary expenses"
#         })
    
#     # If there are very small categories (less than 5% each) that together account for over 20%
#     small_categories = [(cat, amt) for cat, amt in data.items() if amt / total_spending < 0.05]
#     small_total = sum(amt for _, amt in small_categories)
    
#     if small_total / total_spending > 0.2:
#         categories = [cat for cat, _ in small_categories]
#         recommendations.append({
#             "type": "consolidate",
#             "categories": categories,
#             "message": "Consider consolidating spending in smaller categories to better track your expenses"
#         })
    
#     # If there are fewer than 3 categories and total spending is significant
#     if len(data) < 3:
#         recommendations.append({
#             "type": "diversify",
#             "message": "Consider diversifying your discretionary spending across more categories for better balance"
#         })
    
#     return recommendations

# def main():
#     # Set the path for the input CSV file
#     csv_file_path = r"C:\Users\Admin\Desktop\Banking_chatbot\caterlyAI\Fintech\synthetic_transaction_data.csv"  # Replace with the actual path to your CSV file

#     # Set the path for the output JSON file
#     json_output_path = r"C:\Users\Admin\Desktop\Banking_chatbot\caterlyAI\Fintech\output\output.json"  # Replace with the desired path for your JSON output

#     # Load data from the CSV file
#     print(f"Loading data from {csv_file_path}")
#     transactions_df = pd.read_csv(csv_file_path)

#     # Process the data (assuming a function process_data exists in your script)
#     processed_data = BankingDatasetGenerator(transactions_df)

#     # Save processed data to JSON
#     print(f"Saving processed data to {json_output_path}")
#     with open(json_output_path, 'w') as json_file:
#         json.dump(processed_data, json_file, indent=4)

#     print("Data processing complete.")

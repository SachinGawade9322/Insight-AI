import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dateparser
from rapidfuzz import process, fuzz
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import re 
import os
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MockLLM:
    """Mock LLM for fallback when real models fail to load"""
    def _call_(self, *args, **kwargs):
        return '{"companies": [], "categories": [], "months": [], "transaction_type": null, "time_period": null, "start_date": null, "end_date": null}'

class FinancialQueryParser:
    """
    An enhanced class to parse natural language financial queries into structured parameters.
    Handles simple, medium, and complex queries with multiple filters, prioritizing a defined category list.
    """

    def __init__(self, data_path: str):
        """
        Initialize the parser with transaction data and predefined categories.

        Args:
            data_path: Path to the transaction data CSV.
        """
        self.data_path = data_path
        self.transactions_df = None

        self.defined_categories = [
            "Deposit & Addition", "FEES", "OTHER WITHDRAWALS",
            "ELECTRONIC WITHDRAWALS", "ATM & DEBIT CARD WITHDRAWALS",
            "Investment", "Business Expenses", "Personal Care", "Pets",
            "Taxes & Legal", "Charity & Gifts", "Housing", "Education",
            "Subscription", "Travel", "Food & Dining", "Healthcare",
            "Entertainment", "Transport", "Shopping & Retail", "Bills & Utilities"
        ]
        self.categories = list(self.defined_categories)

        self.companies = []
        self.months = ["january", "february", "march", "april", "may", "june",
                      "july", "august", "september", "october", "november", "december"]
        self.company_aliases = {}
        self.category_keywords = self._build_category_keywords() 

        self.load_transaction_data()

        self.llm = self._initialize_model()

        self.parsing_chain = self._create_parsing_chain()

    # *** NEW HELPER METHOD: Build category keywords ***
    def _build_category_keywords(self) -> Dict[str, List[str]]:
        """Builds a keyword map based on self.categories"""
        keywords_map = {}
        # Ensure self.categories exists and is iterable
        if not hasattr(self, 'categories') or not self.categories:
             return {}

        for category in self.categories:
            if not isinstance(category, str): 
                continue

            base_keywords = [
                word.lower() for part in category.split('&')
                for word in part.split()
                if len(word) > 2
            ]
            # Add common variations/synonyms manually
            specific_keywords = {
                "Deposit & Addition": ["deposit", "addition", "income", "salary", "paycheck", "credit"],
                "FEES": ["fee", "charge", "bank fee"],
                "OTHER WITHDRAWALS": ["other withdrawal", "misc withdrawal", "miscellaneous"],
                "ELECTRONIC WITHDRAWALS": ["electronic withdrawal", "transfer", "payment", "eft", "ach"],
                "ATM & DEBIT CARD WITHDRAWALS": ["atm", "debit card", "cash withdrawal"],
                "Investment": ["investment", "stocks", "brokerage", "shares"],
                "Business Expenses": ["business", "work expense"],
                "Personal Care": ["personal care", "hygiene", "salon", "haircut", "cosmetics"],
                "Pets": ["pet", "animal", "vet", "pet food"],
                "Taxes & Legal": ["tax", "legal", "irs", "lawyer"],
                "Charity & Gifts": ["charity", "donation", "gift", "giving"],
                "Housing": ["rent", "mortgage", "housing", "shelter"], 
                "Education": ["education", "school", "tuition", "books", "college", "university"],
                "Subscription": ["subscription", "membership", "recurring", "netflix", "spotify", "hulu"],
                "Travel": ["travel", "trip", "vacation", "flights", "hotel", "airbnb", "transportation"], 
                "Food & Dining": ["food", "dining", "restaurant", "groceries", "coffee", "eat", "meal"],
                "Healthcare": ["health", "medical", "pharmacy", "doctor", "hospital", "clinic", "dental", "vision"],
                "Entertainment": ["entertainment", "movie", "show", "concert", "game", "fun", "streaming"],
                "Transport": ["transport", "transportation", "gas", "fuel", "car", "auto", "parking", "uber", "lyft", "taxi", "bus", "train", "transit"],
                "Shopping & Retail": ["shopping", "retail", "store", "purchase", "buy", "goods", "amazon", "walmart", "target"],
                "Bills & Utilities": ["bill", "utility", "phone", "internet", "electricity", "water", "power", "hydro"] 
            }
            # Combine base keywords and specific keywords, ensure uniqueness
            combined_keywords = list(set(base_keywords + specific_keywords.get(category, [])))
            keywords_map[category] = combined_keywords

        return keywords_map

    def load_transaction_data(self):
        """Load transaction data and merge categories/companies with the predefined lists"""
        data_companies = []
        data_categories = []

        if not os.path.exists(self.data_path):
            print(f"Warning: File '{self.data_path}' not found. Using predefined categories and default aliases only.")
            self.transactions_df = pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Category']) 
            if not self.companies:
                self.company_aliases = {
                    'sbux': 'Starbucks', 'amzn': 'Amazon', 'wf': 'Whole Foods',
                    'tgt': 'Target', 'wmt': 'Walmart', 'costco': 'Costco',
                    'nflx': 'Netflix', 'aapl': 'Apple', 'msft': 'Microsoft'
                }
                self.companies = list(set(self.company_aliases.values()))
        else:
            try:
                self.transactions_df = pd.read_csv(self.data_path)

                # --- Date handling ---
                date_columns = [col for col in self.transactions_df.columns if 'date' in col.lower()]
                if date_columns:
                    date_col = date_columns[0]
                    try:
                        self.transactions_df[date_col] = pd.to_datetime(self.transactions_df[date_col], errors='coerce')
                    except Exception as e:
                        print(f"Warning: Error converting date column '{date_col}': {e}")

                # --- Company handling ---
                company_cols = [col for col in self.transactions_df.columns if any(term in col.lower() for term in ['company', 'merchant', 'vendor', 'payee'])]
                if company_cols:
                    company_col = company_cols[0]
                    self.transactions_df[company_col] = self.transactions_df[company_col].astype(str).str.strip() #
                    data_companies = [c for c in self.transactions_df[company_col].dropna().unique()
                                     if c and c.strip() != '']

                # --- Category handling ---
                category_cols = [col for col in self.transactions_df.columns if any(term in col.lower() for term in ['category', 'type', 'classification'])]
                if category_cols:
                    category_col = category_cols[0]
                    self.transactions_df[category_col] = self.transactions_df[category_col].astype(str).str.strip() 
                    data_categories = [c for c in self.transactions_df[category_col].dropna().unique()
                                      if c and c.strip() != '']

                print(f"Loaded {len(self.transactions_df)} transactions from {self.data_path}")

            except Exception as e:
                print(f"Error loading transaction data from '{self.data_path}': {e}. Using predefined lists only.")
                self.transactions_df = pd.DataFrame() 


        self.companies = list(set(self.companies + data_companies))

        self.company_aliases = {} 
        for company in self.companies:
            if pd.notna(company) and company.strip() != '':
                comp_lower = company.lower()
                self.company_aliases[comp_lower] = company
                words = company.split()
                if len(words) > 1:
                    abbr = ''.join([w[0].lower() for w in words if w])
                    if len(abbr) >= 2 and abbr not in self.company_aliases: 
                        self.company_aliases[abbr] = company

        # Add common abbreviations manually (ensure they map to companies actually present)
        common_abbr = {
            'sbux': 'Starbucks', 'amzn': 'Amazon', 'wf': 'Whole Foods',
            'tgt': 'Target', 'wmt': 'Walmart', 'costco': 'Costco',
            'nflx': 'Netflix', 'aapl': 'Apple', 'msft': 'Microsoft',
            'fb': 'Facebook', 'amex': 'American Express', 'bofa': 'Bank of America',
            'gs': 'Goldman Sachs', 'jpmc': 'JPMorgan Chase', 'tmo': 'T-Mobile',
            'vzw': 'Verizon', 'att': 'AT&T', 'uber': 'Uber', 'lyft': 'Lyft'
        }
        # Add only if the target company exists and the alias isn't already taken
        for abbr, company in common_abbr.items():
            if company in self.companies and abbr not in self.company_aliases:
                 self.company_aliases[abbr] = company


        combined_categories = set(self.defined_categories) | set(data_categories)
        self.categories = sorted(list(combined_categories)) 
        self.category_keywords = self._build_category_keywords()

        print(f"Final setup: {len(self.companies)} companies, {len(self.categories)} categories.")
        # print("Categories:", self.categories) # Uncomment for debugging
        # print("Keywords:", json.dumps(self.category_keywords, indent=2)) # Uncomment for debugging
        # print("Aliases:", self.company_aliases) # Uncomment for debugging

    # *** Initialization of LLM (unchanged, includes fallback) ***
    def _initialize_model(self):
        """Initialize a HuggingFace model for text processing"""
        try:
            model_id = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Initializing LLM on device: {device}")
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
            hf_pipeline = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                max_new_tokens=250, temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.eos_token_id 
            )
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            try:
                 test_output = llm.invoke("test")
                 print("LLM initialized successfully.")
                 return llm
            except Exception as test_e:
                 print(f"LLM test call failed: {test_e}. Falling back.")
                 raise test_e # Raise to trigger fallback

        except Exception as e:
            print(f"Error initializing main LLM ({model_id}): {e}. Trying fallback.")
            try:
                model_id = "distilgpt2"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.pad_token = tokenizer.eos_token
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Initializing Fallback LLM on device: {device}")
                model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
                hf_pipeline = pipeline(
                    "text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=100, pad_token_id=tokenizer.eos_token_id
                )
                llm = HuggingFacePipeline(pipeline=hf_pipeline)
                # Test call for fallback
                try:
                    test_output = llm.invoke("test")
                    print("Fallback LLM initialized successfully.")
                    return llm
                except Exception as test_e2:
                    print(f"Fallback LLM test call failed: {test_e2}. Using Mock LLM.")
                    return MockLLM() # Final fallback: Mock
            except Exception as e2:
                print(f"Error initializing fallback LLM ({model_id}): {e2}. Using Mock LLM.")
                return MockLLM() # Final fallback: Mock

    # *** LLM Parsing Chain Creation (unchanged) ***
    def _create_parsing_chain(self):
        """Create a LangChain chain for parsing queries"""
        template = """
        You are a financial query parser that extracts structured information from natural language queries about financial transactions.

        Given the following user query: "{query}"

        Available Companies: {company_examples} (examples only)
        Available Categories: {category_examples} (examples only)
        Available Months: {month_examples}

        Extract the following information:
        1. Company/Merchant names mentioned (list of strings).
        2. Category names mentioned (list of strings, use names from examples if possible).
        3. Month names mentioned (list of strings).
        4. Transaction type: 'credit' (income, deposits) or 'debit' (expenses, withdrawals, payments).
        5. Date range: Extract start_date and end_date in YYYY-MM-DD format if explicitly mentioned.
        6. Time period: Extract relative time periods like 'last week', 'last month', 'last 3 months', 'Q2 2023'.

        OUTPUT FORMAT:
        Return ONLY a valid JSON object with the following structure:
        {{
          "companies": ["extracted company names or empty list"],
          "categories": ["extracted categories or empty list"],
          "months": ["extracted months or empty list"],
          "transaction_type": "credit or debit or null",
          "time_period": "extracted time period or null",
          "start_date": "YYYY-MM-DD or null",
          "end_date": "YYYY-MM-DD or null"
        }}

        Ensure the JSON is valid. All fields must be present, using null or empty lists [] for unspecified fields. Only return the JSON object.
        """
        prompt = PromptTemplate(
            input_variables=["query", "company_examples", "category_examples", "month_examples"],
            template=template
        )
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=StrOutputParser()
        )
        return chain

    # *** Fuzzy Matching for Companies (refined slightly) ***
    def fuzzy_match_company(self, company_text: str) -> Optional[str]:
        """Find the best company match using aliases and fuzzy matching."""
        if not company_text or pd.isna(company_text):
            return None

        text_lower = company_text.lower().strip()
        if not text_lower:
            return None

        # 1. Check aliases (case-insensitive)
        if text_lower in self.company_aliases:
            return self.company_aliases[text_lower]

        # Filter out empty/invalid companies from the main list
        valid_companies = [c for c in self.companies if pd.notna(c) and c.strip() != '']
        if not valid_companies:
            return None # No companies loaded

        # 2. Try fuzzy matching against the full list of valid company names
        try:
            # token_sort_ratio is good for slightly different names or word order
            result = process.extractOne(
                text_lower,
                valid_companies,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=75 # Adjust cutoff as needed (70-80 is reasonable)
            )
            # result is (choice, score, index) or None
            if result and result[1] >= 75:
                return result[0] # Return the matched company name
            else:
                # Optional: Try partial ratio for substrings? Maybe too broad.
                # result_partial = process.extractOne(text_lower, valid_companies, scorer=fuzz.partial_ratio, score_cutoff=85)
                # if result_partial and result_partial[1] >= 85: return result_partial[0]
                return None
        except Exception as e:
            print(f"Error in fuzzy matching company '{company_text}': {e}")
            return None

    # *** REFINED Fuzzy Matching for Categories ***
    def fuzzy_match_category(self, category_text: str) -> Optional[str]:
        """Find the best category match using exact match, keyword mapping, then fuzzy matching."""
        if not category_text or not self.categories or pd.isna(category_text):
            return None

        lower_text = category_text.lower().strip()
        if not lower_text:
             return None

        # 1. Check for exact match (case-insensitive) with canonical names
        for category in self.categories:
             # Ensure category is a string before lowercasing
            if isinstance(category, str) and category.lower() == lower_text:
                return category

        # 2. Check keyword mappings robustly
        matched_category = None
        best_keyword_match_score = 0

        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                 # Use word boundaries for more precise keyword matching
                if re.search(r'\b' + re.escape(keyword) + r'\b', lower_text):
                    # Score based on keyword length (longer keywords are more specific)
                    current_score = len(keyword)
                    if current_score > best_keyword_match_score:
                         best_keyword_match_score = current_score
                         matched_category = category
                    # Found a keyword match for this category, potentially continue to find best
                    # break # Remove break to find the best match across all categories/keywords

        if matched_category:
             return matched_category # Return the category linked to the best keyword found

        # 3. If no exact or keyword match, try direct fuzzy matching as a fallback
        try:
            valid_categories = [c for c in self.categories if isinstance(c, str) and c.strip() != '']
            if not valid_categories:
                return None

            result = process.extractOne(
                category_text, # Use original case text for potentially better matching? Or lower_text? Test.
                valid_categories,
                scorer=fuzz.token_sort_ratio, # Good for matching phrases
                score_cutoff=75 # Threshold for fuzzy match
            )

            if result and result[1] >= 75:
                return result[0] # Return the matched category name
            return None # No good fuzzy match found
        except Exception as e:
            print(f"Error in fuzzy matching category '{category_text}': {e}")
            return None

    # *** Month Extraction (unchanged) ***
    def extract_month(self, query: str) -> List[str]:
        """Extract month names from query"""
        found_months = []
        query_lower = query.lower()
        # Use word boundaries to avoid matching 'may' in 'maybe'
        for month in self.months:
            if re.search(r'\b' + month + r'\b', query_lower):
                found_months.append(month.capitalize())
        return found_months

    # *** Date Range Inference (unchanged) ***
    def infer_date_range(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Infer date range from relative time periods in query"""
        today = datetime.now()
        query_lower = query.lower()
        start_date, end_date = None, None

        # Use word boundaries for more specific matching
        if re.search(r'\blast week\b', query_lower):
            start_of_last_week = today - timedelta(days=today.weekday() + 7)
            end_of_last_week = start_of_last_week + timedelta(days=6)
            start_date, end_date = start_of_last_week, end_of_last_week
        elif re.search(r'\blast month\b', query_lower):
            end_of_last_month = today.replace(day=1) - timedelta(days=1)
            start_of_last_month = end_of_last_month.replace(day=1)
            start_date, end_date = start_of_last_month, end_of_last_month
        elif re.search(r'\blast 3 months\b', query_lower):
            end_of_period = today.replace(day=1) - timedelta(days=1) # End of last month
            start_of_period = (end_of_period.replace(day=1) - timedelta(days=60)).replace(day=1) # Approx 3 months back
            start_date, end_date = start_of_period, end_of_period
        elif re.search(r'\bthis month\b', query_lower):
            start_of_this_month = today.replace(day=1)
            start_date, end_date = start_of_this_month, today # To today
        elif re.search(r'\bthis year\b', query_lower):
             start_of_this_year = today.replace(month=1, day=1)
             start_date, end_date = start_of_this_year, today
        elif re.search(r'\blast year\b', query_lower):
            start_of_last_year = today.replace(year=today.year - 1, month=1, day=1)
            end_of_last_year = today.replace(year=today.year - 1, month=12, day=31)
            start_date, end_date = start_of_last_year, end_of_last_year
        # Add more specific periods like Quarter (Q1, Q2 etc.) if needed
        # Example: Q2 2023
        q_match = re.search(r'\b(q[1-4])\s+(\d{4})\b', query_lower)
        if q_match:
             quarter, year = q_match.groups()
             year = int(year)
             q_map = {'q1': (1, 3), 'q2': (4, 6), 'q3': (7, 9), 'q4': (10, 12)}
             start_month, end_month = q_map[quarter]
             start_dt = datetime(year, start_month, 1)
             end_day = (datetime(year, end_month + 1, 1) - timedelta(days=1)).day if end_month < 12 else 31
             end_dt = datetime(year, end_month, end_day)
             start_date, end_date = start_dt, end_dt

        # Try parsing explicit dates using dateparser as a fallback
        if start_date is None and end_date is None:
            try:
                # Use prefer_dates_from='past' to resolve ambiguity like "March 5th"
                date_info = dateparser.search.search_dates(query, settings={'PREFER_DATES_FROM': 'past', 'RETURN_AS_TIMEZONE_AWARE': False})
                if date_info:
                    # Find the earliest and latest date found
                    all_dates = [d[1] for d in date_info]
                    if all_dates:
                        min_date = min(all_dates)
                        max_date = max(all_dates)
                        # If only one date found, maybe assume it's start and end? Or just start? Depends on desired behavior.
                        # Let's assume if range isn't clear, we use min/max found.
                        start_date = min_date
                        end_date = max_date if max_date != min_date else None # If only one date, maybe just set start?
                        # If end date is None, maybe default to today or end of month of start_date?
                        # For now, just assign if found.
            except Exception as e:
                # print(f"Dateparser error: {e}") # Uncomment for debugging
                pass # Ignore errors from dateparser

        # Format dates if found
        start_str = start_date.strftime('%Y-%m-%d') if start_date else None
        end_str = end_date.strftime('%Y-%m-%d') if end_date else None

        return start_str, end_str


    # *** Refinement of LLM Output (incorporates fuzzy matching) ***
    def refine_llm_output(self, parsed_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Refine the LLM output with fuzzy matching and rule-based logic where LLM might fail."""
        if not isinstance(parsed_data, dict): # Handle case where LLM output is not a dict
            print("Warning: LLM output was not a valid dictionary.")
            return { "companies": [], "categories": [], "months": [], "transaction_type": None,
                     "time_period": None, "start_date": None, "end_date": None }

        refined = {
            "companies": set(), # Use sets to handle duplicates easily
            "categories": set(),
            "months": set(m.capitalize() for m in parsed_data.get("months", []) if isinstance(m, str) and m.lower() in self.months),
            "transaction_type": parsed_data.get("transaction_type"),
            "time_period": parsed_data.get("time_period"),
            "start_date": parsed_data.get("start_date"),
            "end_date": parsed_data.get("end_date")
        }

        # Fuzzy match companies from LLM output
        llm_companies = parsed_data.get("companies", [])
        if isinstance(llm_companies, list):
            for company_text in llm_companies:
                if isinstance(company_text, str):
                    matched = self.fuzzy_match_company(company_text)
                    if matched:
                        refined["companies"].add(matched)

        # Fuzzy match categories from LLM output
        llm_categories = parsed_data.get("categories", [])
        if isinstance(llm_categories, list):
            for category_text in llm_categories:
                 if isinstance(category_text, str):
                    matched = self.fuzzy_match_category(category_text)
                    if matched:
                        refined["categories"].add(matched)

        # Extract months using rules if LLM missed them
        if not refined["months"]:
            refined["months"].update(self.extract_month(query))

        # Infer date range using rules if LLM missed them or provided only time_period
        # Only infer if BOTH start and end dates are missing from LLM
        if not refined["start_date"] and not refined["end_date"]:
             start_date_inf, end_date_inf = self.infer_date_range(query)
             refined["start_date"] = start_date_inf # Overwrite only if rules found something
             refined["end_date"] = end_date_inf

        # Validate transaction type
        if refined["transaction_type"] not in ["credit", "debit"]:
             refined["transaction_type"] = None # Reset if invalid value

        # Convert sets back to lists for final JSON output
        refined["companies"] = list(refined["companies"])
        refined["categories"] = list(refined["categories"])
        refined["months"] = list(refined["months"])

        return refined

    # *** Main Parsing Function (incorporates rule-based, LLM, refinement) ***
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query using a hybrid rule-based and LLM approach.
        """
        # Default structure
        final_result = {
            "companies": [], "categories": [], "months": [], "transaction_type": None,
            "time_period": None, "start_date": None, "end_date": None
        }
        rule_based_results = final_result.copy() # Start with empty structure
        llm_parsed_refined = None

        try:
            query_lower = query.lower() # Lowercase once

            # --- Step 1: Rule-Based Extraction ---

            # 1a. Direct Company Detection (using aliases and fuzzy match)
            found_companies = set()
            # Simple split by spaces and common delimiters, then check each word/phrase
            potential_company_terms = re.split(r'\s+and\s+|\s*,\s*|\s+', query) # Split query into potential terms
            for term in potential_company_terms:
                 term = term.strip()
                 if len(term) > 1: # Avoid single letters
                    matched = self.fuzzy_match_company(term) # Use fuzzy match here directly
                    if matched:
                        found_companies.add(matched)
            rule_based_results["companies"] = list(found_companies)

            # 1b. Category Detection (using exact match and keywords)
            found_categories = set()
            # Check for exact matches first (case-insensitive, whole word)
            for category in self.categories:
                if isinstance(category, str): # Ensure it's a string
                     # Use word boundaries for exact category name matching
                    if re.search(r'\b' + re.escape(category.lower()) + r'\b', query_lower):
                        found_categories.add(category)

            # Check using the keyword map
            for category, keywords in self.category_keywords.items():
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                        found_categories.add(category)
                        break # Found a keyword for this category, move to next category

            rule_based_results["categories"] = list(found_categories)


            # 1c. Month Detection
            rule_based_results["months"] = self.extract_month(query) # Uses query, not query_lower internally

            # 1d. Transaction Type Detection
            debit_keywords = ["spend", "spent", "expense", "purchase", "bought", "paid", "pay", "debit", "cost", "expenses", "withdrawal", "charge", "fee"]
            credit_keywords = ["income", "earn", "received", "deposit", "credit", "salary", "wage", "payment received", "revenue", "addition"]
            rule_based_results["transaction_type"] = None # Reset before checking
            # Check debit first
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', query_lower) for keyword in debit_keywords):
                 rule_based_results["transaction_type"] = "debit"
            # Check credit if debit not found or ambiguous
            if rule_based_results["transaction_type"] is None:
                if any(re.search(r'\b' + re.escape(keyword) + r'\b', query_lower) for keyword in credit_keywords):
                    rule_based_results["transaction_type"] = "credit"

            # Infer type from categories if still None
            if rule_based_results["transaction_type"] is None:
                 if any(cat in rule_based_results["categories"] for cat in ["FEES", "OTHER WITHDRAWALS", "ELECTRONIC WITHDRAWALS", "ATM & DEBIT CARD WITHDRAWALS", "Business Expenses", "Personal Care", "Pets", "Taxes & Legal", "Charity & Gifts", "Housing", "Education", "Subscription", "Travel", "Food & Dining", "Healthcare", "Entertainment", "Transport", "Shopping & Retail", "Bills & Utilities", "Investment"]):
                      rule_based_results["transaction_type"] = "debit" # Most categories imply debit
                 elif "Deposit & Addition" in rule_based_results["categories"]:
                      rule_based_results["transaction_type"] = "credit"

            time_periods = ["last week", "last month", "last year", "this week",
                          "this month", "this year", "past week", "past month",
                          "past year", "last 3 months", "past 6 months", "last quarter",
                          r'q[1-4]\s+\d{4}'] # Regex for Q1 2023 etc.

            for period_pattern in time_periods:
                match = re.search(r'\b' + period_pattern + r'\b', query_lower, re.IGNORECASE)
                if match:
                    rule_based_results["time_period"] = match.group(0).strip() 
                    break 

            # Infer date range based on query (handles time periods and explicit dates)
            start_date, end_date = self.infer_date_range(query)
            rule_based_results["start_date"] = start_date
            rule_based_results["end_date"] = end_date

            needs_llm = not (rule_based_results["companies"] and rule_based_results["categories"]) \
                        or len(query.split()) > 10 # Simple complexity check

            if needs_llm:
                print("Rule-based extraction incomplete or query complex, invoking LLM...")
                try:
                    # Prepare examples for the prompt
                    company_examples = ", ".join(self.companies[:5]) if self.companies else "Amazon, Starbucks"
                    category_examples = ", ".join(self.categories[:5]) if self.categories else "Food, Shopping"
                    month_examples = ", ".join(self.months[:3])

                    llm_output_raw = self.parsing_chain.invoke({
                        "query": query,
                        "company_examples": company_examples,
                        "category_examples": category_examples,
                        "month_examples": month_examples
                    })

                    # Clean and parse LLM output
                    json_match = re.search(r'\{.*\}', llm_output_raw, re.DOTALL)
                    if json_match:
                        llm_output_json = json_match.group(0)
                        try:
                            llm_parsed_raw = json.loads(llm_output_json)
                            llm_parsed_refined = self.refine_llm_output(llm_parsed_raw, query)
                        except json.JSONDecodeError as json_e:
                            print(f"Error decoding LLM JSON: {json_e}")
                            print(f"LLM Raw Output:\n{llm_output_raw}")
                    else:
                        print("LLM did not return valid JSON structure.")
                        print(f"LLM Raw Output:\n{llm_output_raw}")

                except Exception as llm_e:
                    print(f"Error during LLM parsing/invocation: {llm_e}")

            final_result = rule_based_results.copy()

            if llm_parsed_refined:
                # Merge lists (companies, categories, months) - take union
                for key in ["companies", "categories", "months"]:
                    final_result[key] = sorted(list(set(final_result[key]) | set(llm_parsed_refined.get(key, []))))

                # Prefer LLM for single-value fields IF rule-based missed them
                for key in ["transaction_type", "time_period", "start_date", "end_date"]:
                    if final_result[key] is None and llm_parsed_refined.get(key) is not None:
                        final_result[key] = llm_parsed_refined[key]
                    if key in ["start_date", "end_date"] and final_result[key] is None and llm_parsed_refined.get(key):
                         final_result[key] = llm_parsed_refined[key]
                    elif key == "time_period" and final_result[key] is None and llm_parsed_refined.get(key):
                         final_result[key] = llm_parsed_refined[key]

            return final_result

        except Exception as e:
            print(f"Critical error parsing query '{query}': {e}")
            # Return the default empty structure in case of unexpected errors
            return {
                "companies": [], "categories": [], "months": [], "transaction_type": None,
                "time_period": None, "start_date": None, "end_date": None
            }

if __name__ == "__main__":
    import sys

    print("📊 Financial Query Parser - CLI Mode")
    print("Type your query below. Type 'exit' or press Ctrl+C to quit.\n")

    # Initialize the parser
    parser = FinancialQueryParser(r"C:\Users\Admin\Desktop\Banking_chatbot\caterlyAI\Fintech\modified_test.csv")

    while True:
        try:
            user_query = input("🗣️ Enter query: ").strip()
            if user_query.lower() in ["exit", "quit"]:
                print("👋 Exiting parser. Goodbye!")
                break

            parsed = parser.parse_query(user_query)
            print("✅ Parsed result:")
            print(json.dumps(parsed, indent=2))

        except KeyboardInterrupt:
            print("\n👋 Exiting parser. Goodbye!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

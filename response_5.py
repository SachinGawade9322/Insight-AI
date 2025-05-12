import os
import json
import torch
import pandas as pd
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional
from promt_engine_3 import FinancialQueryParser
from memory_manager import ChatMemoryManager
from statement_memory_manager import BankStatementMemoryManager

MODEL_DIR = "./fintech_model_output"
TRANSACTION_DATA_PATH = r"C:\Users\Admin\Desktop\Banking_chatbot\caterlyAI\Fintech\modified_test.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FinTechChatbot:
    def __init__(self, model_dir: str, data_path: str):
        self.model_dir = model_dir
        self.data_path = data_path
        self.model = None
        self.tokenizer = None
        self.transactions_df = None
        self.query_parser = FinancialQueryParser(data_path)
        self.session_id = str(uuid.uuid4())
        self.chat_memory_manager = ChatMemoryManager(user_id="system_default_user", k=5)
        self.statement_memory_manager = BankStatementMemoryManager(user_id="system_default_user")
        #self.chat_memory_manager.initialize_memory(self.session_id)
        self.load_resources()

    def load_resources(self):
        print("Loading resources...")
        try:
            self.transactions_df = pd.read_csv(self.data_path)
            
            if 'Amount' in self.transactions_df.columns:
                self.transactions_df['Amount'] = pd.to_numeric(
                    self.transactions_df['Amount'], errors='coerce'
                )

            if 'Type' in self.transactions_df.columns:
                type_mapping = {
                    'credit': 'credit', 'deposit': 'credit', 'income': 'credit', 'salary': 'credit',
                    'debit': 'debit', 'withdrawal': 'debit', 'expense': 'debit', 'payment': 'debit'
                }
                for key, value in type_mapping.items():
                    mask = self.transactions_df['Type'].str.contains(key, case=False, na=False)
                    self.transactions_df.loc[mask, 'Type'] = value

            print(f"Loaded {len(self.transactions_df)} transactions")
        except Exception as e:
            print(f"Error loading transaction data: {e}")
            raise

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

        try:
            from transformers import BitsAndBytesConfig
            torch.cuda.empty_cache()

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="./offload"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def merge_params(self, previous: Optional[Dict], current: Dict) -> Dict:
        """Smart merging of current and historical filters"""
        merged = previous.copy() if previous else {}
        for key in ['companies', 'categories', 'months']:
            if current.get(key):
                merged[key] = list(set(merged.get(key, []) + current[key]))
        for key in ['transaction_type', 'time_period', 'start_date', 'end_date']:
            if current.get(key):
                merged[key] = current[key]
        return merged

    def filter_transactions(self, params: Dict[str, Any]) -> pd.DataFrame:
        filtered_df = self.transactions_df.copy()

        if params.get("companies"):
            companies = [c.lower() for c in params["companies"]]
            filtered_df = filtered_df[filtered_df['Company'].str.lower().isin(companies)]

        if params.get("categories"):
            categories = [c.lower() for c in params["categories"]]
            filtered_df = filtered_df[filtered_df['Category'].str.lower().isin(categories)]

        if params.get("transaction_type"):
            t_type = params["transaction_type"].lower()
            filtered_df = filtered_df[filtered_df['Type'].str.contains(t_type, case=False, na=False)]

        if params.get("start_date") or params.get("end_date"):
            date_col = next((col for col in filtered_df.columns if 'date' in col.lower()), None)
            if date_col:
                if params["start_date"]:
                    filtered_df = filtered_df[filtered_df[date_col] >= params["start_date"]]
                if params["end_date"]:
                    filtered_df = filtered_df[filtered_df[date_col] <= params["end_date"]]

        if 'Date' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='Date', ascending=False)

        return filtered_df

    def format_transactions_for_prompt(self, transactions_df: pd.DataFrame, params: Dict[str, Any], limit: int = 20) -> str:
        if transactions_df.empty:
            return "No transactions found matching the criteria."

        if len(transactions_df) > limit:
            transactions_df = transactions_df.head(limit)

        transactions_text = "Relevant transactions:\n\n"
        for idx, row in transactions_df.iterrows():
            date_str = str(row['Date']) if 'Date' in row else "N/A"
            amount = float(row['Amount']) if 'Amount' in row else 0.0
            amount_str = f"${abs(amount):.2f}"

            transactions_text += f"""Transaction #{idx+1}:
Date: {date_str}
Amount: {amount_str}
Type: {row.get('Type', 'N/A')}
Description: {row.get('Description', 'N/A')}
Category: {row.get('Category', 'N/A')}
Company: {row.get('Company', 'N/A')}\n\n"""

        if not transactions_df.empty and 'Amount' in transactions_df.columns:
            total_amount = transactions_df['Amount'].abs().sum()
            avg_amount = transactions_df['Amount'].abs().mean()

            transactions_text += f"Summary Statistics:\n"
            transactions_text += f"Total Transactions: {len(transactions_df)}\n"
            transactions_text += f"Total Amount: ${total_amount:.2f}\n"
            transactions_text += f"Average Amount: ${avg_amount:.2f}\n"

        return transactions_text

    def extract_query_intent(self, query: str) -> str:
        """Extract the intent behind a user query"""
        query_lower = query.lower()
        
        # Check for confirmation/surprise
        if any(word in query_lower for word in ["really", "actually", "seriously", "wow"]):
            return "confirmation"
        
        # Check for evaluation requests
        if any(word in query_lower for word in ["too much", "enough", "right", "okay", "guess"]):
            return "evaluation"
        
        # Check for advice requests
        if any(phrase in query_lower for phrase in ["what can i do", "how can i", "should i", "save", "reduce"]):
            return "advice"
        
        # Default to information request
        return "information"

    def enhance_query_with_context(self, query: str, intent: str, is_follow_up: bool) -> str:
        """Enhance the query with context about what the user is really asking"""
        if not is_follow_up:
            return query
            
        if intent == "confirmation":
            return f"{query} Please confirm the amounts and provide more context or breakdown."
        elif intent == "evaluation":
            return f"{query} Evaluate if this spending is appropriate and provide benchmarks or comparisons."
        elif intent == "advice":
            return f"{query} Provide specific actionable advice on how to reduce costs or save money in this area."
        else:
            return query

    def generate_ai_insights(self, transactions_df: pd.DataFrame, query: str, params: Dict[str, Any], query_intent: str = "information") -> str:
        transactions_text = self.format_transactions_for_prompt(transactions_df, params)

        context = []
        if params.get("companies"):
            context.append(f"Focus on transactions with companies: {', '.join(params['companies'])}")
        if params.get("categories"):
            context.append(f"Focus on transactions in categories: {', '.join(params['categories'])}")
        if params.get("transaction_type"):
            context.append(f"Focus on {params['transaction_type']} transactions")

        # Customize response format based on query intent
        prompt_instructions = """Provide analysis with:
1. Transaction breakdown
2. Spending/income patterns
3. Category insights
4. Actionable recommendations
5. Key summary"""

        if query_intent == "confirmation":
            prompt_instructions = """The user is asking to confirm their spending amounts. Provide:
1. Clear confirmation of the amounts
2. More detailed breakdown of relevant transactions 
3. Additional context about these transactions
4. Brief comment on whether this amount is significant relative to their overall spending
5. Brief summary that directly answers their question about the spending amount"""
        
        elif query_intent == "evaluation":
            prompt_instructions = """The user is asking for your evaluation of their spending. Provide:
1. Brief transaction recap
2. Evaluation of whether the spending seems high, normal, or low
3. Comparison to typical spending in this category
4. Your assessment of whether they're spending too much
5. Brief recommendations"""
            
        elif query_intent == "advice":
            prompt_instructions = """The user is asking for specific advice to reduce costs. Provide:
1. Very brief transaction recap
2. Specific, actionable steps to reduce spending in this category
3. Potential savings amounts for each recommendation
4. Alternative products or services to consider
5. Long-term strategies for continued savings"""

        prompt = f"""<s>[INST] You are a financial expert chatbot named FinTech. Analyze transactions and provide insights for: "{query}"

{transactions_text}

{" ".join(context)}

{prompt_instructions}
[/INST]"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1].strip()

    def process_query(self, query: str) -> str:
        try:
            print("Rule-based extraction incomplete or query complex, invoking LLM...")
            
            current_statement_df = self.statement_memory_manager.get_bank_statement(self.session_id)
            if current_statement_df is not None:
                self.transactions_df = current_statement_df

            # Extract the intent from the query
            query_intent = self.extract_query_intent(query)
            
            parsed_params = self.query_parser.parse_query(query)
            
            context = self.chat_memory_manager.get_context(self.session_id)
            
            if context and len(context) == 2:
                previous_params, previous_transactions = context
            else:
                previous_params, previous_transactions = None, None
                
            merged_params = self.merge_params(previous_params, parsed_params)

            # Enhanced follow-up detection
            follow_up_triggers = {
                'these', 'this', 'that', 'above', 'earlier', 'those', 
                'much', 'really', 'okay', 'right', 'guess', 'reduce', 
                'save', 'spending'
            }
            
            is_follow_up = (
                any(trigger in query.lower() for trigger in follow_up_triggers) or 
                len(query.split()) <= 6
            )
            
            # For follow-up questions, handle differently based on intent
            if is_follow_up and previous_params:
                if not any(merged_params.get(k) for k in ['companies', 'categories', 'transaction_type']):
                    merged_params.update({k: v for k,v in previous_params.items() if k in ['companies', 'categories']})
                
                # Use previous filtered transactions if the query doesn't specify new filters
                if len(parsed_params.keys()) <= 1 and previous_transactions is not None and not previous_transactions.empty:
                    filtered_transactions = previous_transactions
                else:
                    filtered_transactions = self.filter_transactions(merged_params)
            else:
                filtered_transactions = self.filter_transactions(merged_params)

            self.chat_memory_manager.update_context(self.session_id, merged_params, filtered_transactions)

            if filtered_transactions.empty:
                return "No matching transactions found based on current context. Please refine your query."

            # Pass the intent to the insight generator
            response = self.generate_ai_insights(
                filtered_transactions, 
                query, 
                merged_params,
                query_intent
            )
            
            self.chat_memory_manager.save_to_memory(self.session_id, query, response)

            return response

        except Exception as e:
            print(f"Error processing query: {e}")
            return f"Error processing your query: {str(e)}. Please try again."

def analyze_csv_structure(filepath):
    try:
        df = pd.read_csv(filepath)
        print("\nCSV File Structure Analysis:")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")

        for col in df.columns:
            unique_values = df[col].dropna().unique()
            sample = unique_values[:5] if len(unique_values) > 5 else unique_values
            print(f"\n{col} - {len(unique_values)} unique values")
            print(f"Sample values: {', '.join(str(x) for x in sample)}")

    except Exception as e:
        print(f"Error analyzing CSV structure: {e}")

def run_chatbot_interface():
    print("=" * 50)
    print("FinTech Chatbot - Transaction Analysis")
    print("=" * 50)
    print("Type 'exit' to quit")
    print("Type 'analyze_csv' to see transaction data structure")
    print("Type 'upload <path_to_file>' to upload a new bank statement")
    print("=" * 50)

    try:
        chatbot = FinTechChatbot(MODEL_DIR, TRANSACTION_DATA_PATH)

        while True:
            user_query = input("\nYour query: ")

            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("Thank you for using FinTech Chatbot!")
                break

            if user_query.lower() == 'analyze_csv':
                analyze_csv_structure(TRANSACTION_DATA_PATH)
                continue

            if user_query.lower().startswith("upload "):
                file_path = user_query.split(" ", 1)[-1].strip()

                if os.path.exists(file_path):
                    try:
                        new_df = pd.read_csv(file_path)
                        chatbot.statement_memory_manager.save_bank_statement(chatbot.session_id, new_df)
                        chatbot.chat_memory_manager.reset_memory(chatbot.session_id)
                        print("✅ Bank statement uploaded and chat memory reset successfully!")
                    except Exception as e:
                        print(f"❌ Error uploading statement: {e}")
                else:
                    print(f"❌ File not found: {file_path}")
                continue

            print("\nAnalyzing your query...")
            response = chatbot.process_query(user_query)

            print("\n" + "=" * 50)
            print("FinTech Assistant:")
            print(response)
            print("=" * 50)

    except KeyboardInterrupt:
        print("\nChatbot session terminated.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    run_chatbot_interface()
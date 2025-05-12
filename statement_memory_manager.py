
# import redis
# import json
# import traceback

# class BankStatementMemoryManager:
#     def __init__(self, user_id, ttl=3600):
#         self.user_id = user_id
#         self.ttl = ttl
#         self.statement_key = f"statement_memory_{user_id}"
#         self.statement_data = ""

#         try:
#             self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
#             self.redis_client.ping()
#             self.connected = True
#         except Exception as e:
#             print("Warning: Could not connect to Redis. Using in-memory fallback.")
#             print("Error:", e)
#             self.redis_client = None
#             self.connected = False

#     def save_statement_memory(self, statement_json_str):
#         try:
#             if self.connected:
#                 self.redis_client.setex(self.statement_key, self.ttl, statement_json_str)
#             else:
#                 self.statement_data = statement_json_str
#         except Exception:
#             print("Error saving statement memory:")
#             traceback.print_exc()

#     def load_statement_memory(self):
#         try:
#             if self.connected:
#                 data = self.redis_client.get(self.statement_key)
#                 if data:
#                     return data.decode("utf-8")
#                 return ""
#             else:
#                 return self.statement_data
#         except Exception:
#             print("Error loading statement memory:")
#             traceback.print_exc()
#             return ""





import redis
import json
import traceback
import pandas as pd

class BankStatementMemoryManager:
    def __init__(self, user_id, ttl=3600):
        self.user_id = user_id
        self.ttl = ttl
        self.statement_key = f"statement_memory_{user_id}"
        self.statement_data = ""

        try:
            self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
            self.redis_client.ping()
            self.connected = True
            print(f"✅ Successfully connected to Redis for bank statement (user_id: {user_id})")
        except Exception as e:
            print("⚠️ Warning: Could not connect to Redis. Using in-memory fallback for bank statement.")
            print(f"Error: {e}")
            self.redis_client = None
            self.connected = False

    def save_statement_memory(self, statement_json_str):
        try:
            if self.connected:
                self.redis_client.setex(self.statement_key, self.ttl, statement_json_str)
            else:
                self.statement_data = statement_json_str
            return True
        except Exception as e:
            print(f"❌ Error saving bank statement: {str(e)}")
            traceback.print_exc()
            return False

    def load_statement_memory(self):
        try:
            if self.connected:
                data = self.redis_client.get(self.statement_key)
                if data:
                    return data.decode("utf-8")
                return ""
            else:
                return self.statement_data
        except Exception as e:
            print(f"❌ Error loading bank statement: {str(e)}")
            traceback.print_exc()
            return ""
    
    # For backward compatibility with the old main.py API
    def get_bank_statement(self, session_id):
        """For backward compatibility - converts JSON to DataFrame"""
        json_data = self.load_statement_memory()
        if json_data:
            try:
                return pd.read_json(json_data)
            except Exception as e:
                print(f"❌ Error converting JSON to DataFrame: {str(e)}")
                return None
        return None
    
    def save_bank_statement(self, session_id, bank_statement_df):
        """For backward compatibility - converts DataFrame to JSON"""
        try:
            json_data = bank_statement_df.to_json(orient="records")
            return self.save_statement_memory(json_data)
        except Exception as e:
            print(f"❌ Error converting DataFrame to JSON: {str(e)}")
            return False
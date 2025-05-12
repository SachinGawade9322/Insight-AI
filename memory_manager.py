
# import redis
# import json
# import traceback

# class ChatMemoryManager:
#     def __init__(self, user_id, ttl=3600):
#         self.user_id = user_id
#         self.ttl = ttl
#         self.memory_key = f"memory_{user_id}"
#         self.memory_data = []

#         try:
#             self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
#             self.redis_client.ping()
#             self.connected = True
#         except Exception as e:
#             print("Warning: Could not connect to Redis. Using in-memory fallback.")
#             print("Error:", e)
#             self.redis_client = None
#             self.connected = False

#     def save_memory(self, messages):
#         try:
#             data = json.dumps(messages)
#             if self.connected:
#                 self.redis_client.setex(self.memory_key, self.ttl, data)
#             else:
#                 self.memory_data = messages
#         except Exception:
#             print("Error saving memory:")
#             traceback.print_exc()

#     def load_memory(self):
#         try:
#             if self.connected:
#                 data = self.redis_client.get(self.memory_key)
#                 if data:
#                     return json.loads(data.decode("utf-8"))
#                 return []
#             else:
#                 return self.memory_data
#         except Exception:
#             print("Error loading memory:")
#             traceback.print_exc()
#             return []




# import redis
# import json
# import traceback

# class ChatMemoryManager:
#     def __init__(self, user_id, ttl=3600, k=5):
#         self.user_id = user_id
#         self.k = k
#         self.ttl = ttl
#         self.memory_key = f"memory_{user_id}"
#         self.memory_data = []

#         try:
#             self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
#             self.redis_client.ping()
#             self.connected = True
#             print(f"✅ Successfully connected to Redis for chat memory (user_id: {user_id})")
#         except Exception as e:
#             print("⚠️ Warning: Could not connect to Redis. Using in-memory fallback for chat memory.")
#             print(f"Error: {e}")
#             self.redis_client = None
#             self.connected = False

#     def save_memory(self, messages):
#         try:
#             data = json.dumps(messages)
#             if self.connected:
#                 self.redis_client.setex(self.memory_key, self.ttl, data)
#             else:
#                 self.memory_data = messages
#             return True
#         except Exception as e:
#             print(f"❌ Error saving chat memory: {str(e)}")
#             traceback.print_exc()
#             return False

#     def load_memory(self):
#         try:
#             if self.connected:
#                 data = self.redis_client.get(self.memory_key)
#                 if data:
#                     return json.loads(data.decode("utf-8"))
#                 return []
#             else:
#                 return self.memory_data
#         except Exception as e:
#             print(f"❌ Error loading chat memory: {str(e)}")
#             traceback.print_exc()
#             return []
            
#     # For backward compatibility with the previous API
#     def get_conversation(self, session_id):
#         """Alias for load_memory to ensure compatibility with main.py"""
#         return self.load_memory()
    
#     def save_conversation(self, session_id, user_query, answer):
#         """Alias for maintaining compatibility with main.py"""
#         messages = self.load_memory()
#         messages.append({"role": "user", "content": user_query})
#         messages.append({"role": "assistant", "content": answer})
#         return self.save_memory(messages)









import redis
import json
import traceback

class ChatMemoryManager:
    def __init__(self, user_id, ttl=3600, k=5):
        self.user_id = user_id
        self.k = k
        self.ttl = ttl
        self.memory_key = f"memory_{user_id}"
        self.memory_data = []

        try:
            self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
            self.redis_client.ping()
            self.connected = True
            print(f"✅ Successfully connected to Redis for chat memory (user_id: {user_id})")
        except Exception as e:
            print("⚠️ Warning: Could not connect to Redis. Using in-memory fallback for chat memory.")
            print(f"Error: {e}")
            self.redis_client = None
            self.connected = False

    def save_memory(self, messages):
        try:
            data = json.dumps(messages)
            if self.connected:
                self.redis_client.setex(self.memory_key, self.ttl, data)
            else:
                self.memory_data = messages
            return True
        except Exception as e:
            print(f"❌ Error saving chat memory: {str(e)}")
            traceback.print_exc()
            return False

    def load_memory(self):
        try:
            if self.connected:
                data = self.redis_client.get(self.memory_key)
                if data:
                    return json.loads(data.decode("utf-8"))
                return []
            else:
                return self.memory_data
        except Exception as e:
            print(f"❌ Error loading chat memory: {str(e)}")
            traceback.print_exc()
            return []
            
    # For backward compatibility with the previous API
    def get_conversation(self, session_id):
        """Alias for load_memory to ensure compatibility with main.py"""
        return self.load_memory()
    
    def save_conversation(self, session_id, user_query, answer):
        """Alias for maintaining compatibility with main.py"""
        messages = self.load_memory()
        messages.append({"role": "user", "content": user_query})
        messages.append({"role": "assistant", "content": answer})
        return self.save_memory(messages)
        
    def get_context(self, session_id=None, k=None):
        """
        Get the last k conversation entries as context.
        If k is not provided, use the default k value from the class.
        """
        if k is None:
            k = self.k
            
        messages = self.load_memory()
        # Return the last k messages (or all if fewer than k)
        return messages[-k:] if messages else []
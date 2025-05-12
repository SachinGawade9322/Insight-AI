# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pandas as pd
# import uuid
# import io

# from memory_manager import ChatMemoryManager
# from statement_memory_manager import BankStatementMemoryManager
# from response_5 import FinTechChatbot

# app = FastAPI(
#     title="💼 Bank Statement Consultant API",
#     description="Upload a bank statement and ask dynamic financial questions.",
#     version="1.0.0"
# )

# # Allow frontend requests from all origins for now
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Generate a default user ID for the application
# DEFAULT_USER_ID = "system_default_user"

# # Initialize Redis-backed managers
# try:
#     memory_manager = ChatMemoryManager(user_id=DEFAULT_USER_ID)
#     # Also fix the BankStatementMemoryManager initialization to match its constructor
#     statement_manager = BankStatementMemoryManager(user_id=DEFAULT_USER_ID)
# except Exception as e:
#     raise HTTPException(status_code=500, detail=f"❌ Redis connection failed. Ensure it's running on localhost. Error: {str(e)}")

# # ----------------------------
# # Data Model for User Queries
# # ----------------------------
# class QueryModel(BaseModel):
#     session_id: str
#     user_query: str

# # ----------------------------
# # Upload Bank Statement
# # ----------------------------
# @app.post("/upload-statement")
# async def upload_statement(session_id: str = Form(...), file: UploadFile = File(...)):
#     if not file.filename.endswith(".csv"):
#         raise HTTPException(status_code=400, detail="Only CSV files are supported.")

#     try:
#         content = await file.read()
#         df = pd.read_csv(io.StringIO(content.decode("utf-8")))
#         # Convert DataFrame to JSON string for BankStatementMemoryManager
#         statement_json = df.to_json(orient="records")
#         statement_manager.save_statement_memory(statement_json)
#         return {"message": "✅ Bank statement uploaded and saved successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"❌ Failed to process CSV file: {e}")

# # ----------------------------
# # Ask Question Based on Bank Statement
# # ----------------------------
# # @app.post("/chat")
# # async def ask_question(data: QueryModel):
# #     session_id = data.session_id
# #     user_query = data.user_query

# #     # Get stored bank statement
# #     statement_json = statement_manager.load_statement_memory()
# #     if not statement_json:
# #         raise HTTPException(status_code=400, detail="⚠ No bank statement found. Please upload one first.")
    
# #     # Convert JSON string back to DataFrame
# #     statement_df = pd.read_json(statement_json)

# #     # Get conversation memory
# #     try:
# #         recent_history = memory_manager.load_memory()
# #         answer = FinTechChatbot(user_query, statement_df, recent_history)
        
# #         # Update memory with new conversation
# #         recent_history.append({"role": "user", "content": user_query})
# #         recent_history.append({"role": "assistant", "content": answer})
# #         memory_manager.save_memory(recent_history)
        
# #         return JSONResponse(content={"response": answer})
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"❌ Error generating response: {e}")

# @app.post("/chat")
# async def chat(request: QueryModel):
#     user_query = request.prompt
#     session_id = request.session_id

#     # Get stored bank statement
#     statement_json = statement_manager.load_statement_memory()
#     if not statement_json:
#         raise HTTPException(status_code=400, detail="⚠ No bank statement found. Please upload one first.")
    
#     # Convert JSON string back to DataFrame
#     try:
#         statement_df = pd.read_json(statement_json)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"❌ Error parsing bank statement data: {e}")

#     # Get conversation memory
#     try:
#         recent_history = memory_manager.load_memory()
#         answer = FinTechChatbot(user_query, statement_df, recent_history)
        
#         # Update memory with new conversation
#         recent_history.append({"role": "user", "content": user_query})
#         recent_history.append({"role": "assistant", "content": answer})
#         memory_manager.save_memory(recent_history)
        
#         return JSONResponse(content={"response": answer})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"❌ Error generating response: {e}")

# # ----------------------------
# # Generate Session ID (Optional)
# # ----------------------------
# @app.get("/session")
# def generate_session():
#     return {"session_id": str(uuid.uuid4())}

# # ----------------------------
# # View Current Statement (Optional)
# # ----------------------------
# @app.get("/statement")
# def get_current_statement(session_id: str):
#     try:
#         statement_json = statement_manager.load_statement_memory()
#         if not statement_json:
#             raise HTTPException(status_code=404, detail="No statement found for this session.")
#         statement_df = pd.read_json(statement_json)
#         return statement_df.to_dict(orient="records")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error retrieving statement: {e}")







# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pandas as pd
# import uuid
# import io

# from memory_manager import ChatMemoryManager
# from statement_memory_manager import BankStatementMemoryManager
# from response_5 import FinTechChatbot

# app = FastAPI(
#     title="💼 Bank Statement Consultant API",
#     description="Upload a bank statement and ask dynamic financial questions.",
#     version="1.0.0"
# )

# # Allow frontend requests from all origins for now
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Generate a default user ID for the application
# DEFAULT_USER_ID = "system_default_user"

# # Initialize Redis-backed managers
# try:
#     memory_manager = ChatMemoryManager(user_id=DEFAULT_USER_ID)
#     # Also fix the BankStatementMemoryManager initialization to match its constructor
#     statement_manager = BankStatementMemoryManager(user_id=DEFAULT_USER_ID)
# except Exception as e:
#     raise HTTPException(status_code=500, detail=f"❌ Redis connection failed. Ensure it's running on localhost. Error: {str(e)}")

# # ----------------------------
# # Data Model for User Queries
# # ----------------------------
# class QueryModel(BaseModel):
#     session_id: str
#     user_query: str

# # ----------------------------
# # Upload Bank Statement
# # ----------------------------
# @app.post("/upload-statement")
# async def upload_statement(session_id: str = Form(...), file: UploadFile = File(...)):
#     if not file.filename.endswith(".csv"):
#         raise HTTPException(status_code=400, detail="Only CSV files are supported.")

#     try:
#         content = await file.read()
#         df = pd.read_csv(io.StringIO(content.decode("utf-8")))
#         # Convert DataFrame to JSON string for BankStatementMemoryManager
#         statement_json = df.to_json(orient="records")
#         statement_manager.save_statement_memory(statement_json)
#         return {"message": "✅ Bank statement uploaded and saved successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"❌ Failed to process CSV file: {e}")

# # ----------------------------
# # Ask Question Based on Bank Statement
# # ----------------------------
# @app.post("/chat")
# async def chat(data: QueryModel):
#     user_query = data.user_query
#     session_id = data.session_id

#     # Get stored bank statement
#     statement_json = statement_manager.load_statement_memory()
#     if not statement_json:
#         raise HTTPException(status_code=400, detail="⚠ No bank statement found. Please upload one first.")
    
#     # Convert JSON string back to DataFrame
#     try:
#         statement_df = pd.read_json(statement_json)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"❌ Error parsing bank statement data: {e}")

#     # Get conversation memory
#     try:
#         # Create a temporary file path for the statement
#         temp_csv_path = "./temp_statement.csv"
#         statement_df.to_csv(temp_csv_path, index=False)
        
#         # Initialize the FinTechChatbot with proper paths
#         model_dir = "./fintech_model_output"  # Adjust as needed
        
#         chatbot = FinTechChatbot(model_dir, temp_csv_path)
        
#         # Override the transactions_df with our loaded statement
#         chatbot.transactions_df = statement_df
        
#         # Load conversation memory
#         recent_history = memory_manager.load_memory()
        
#         # Store the session ID in the chatbot
#         chatbot.session_id = session_id
        
#         # Process the query
#         answer = chatbot.process_query(user_query)
        
#         # Update memory with new conversation
#         recent_history.append({"role": "user", "content": user_query})
#         recent_history.append({"role": "assistant", "content": answer})
#         memory_manager.save_memory(recent_history)
        
#         return JSONResponse(content={"response": answer})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"❌ Error generating response: {e}")
    
# # ----------------------------
# # Generate Session ID (Optional)
# # ----------------------------
# @app.get("/session")
# def generate_session():
#     return {"session_id": str(uuid.uuid4())}

# # ----------------------------
# # View Current Statement (Optional)
# # ----------------------------
# @app.get("/statement")
# def get_current_statement(session_id: str):
#     try:
#         statement_json = statement_manager.load_statement_memory()
#         if not statement_json:
#             raise HTTPException(status_code=404, detail="No statement found for this session.")
#         statement_df = pd.read_json(statement_json)
#         return statement_df.to_dict(orient="records")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error retrieving statement: {e}")
    
    
    
    
    
    
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uuid
import io
import os
import time

from memory_manager import ChatMemoryManager
from statement_memory_manager import BankStatementMemoryManager
from response_5 import FinTechChatbot

# Global variables for model and paths
MODEL_DIR = "./fintech_model_output"  # Adjust path as needed
TEMP_CSV_PATH = "./temp_statement.csv"  # Default path for temporary statement storage
DEFAULT_USER_ID = "system_default_user"

# Initialize the chatbot at module level before FastAPI app startup
chatbot = None

# Create an empty dataframe for initial model loading if no CSV exists yet
def create_empty_dataframe():
    # Create a basic empty dataframe with expected columns
    # Adjust column names as needed based on your model's expectations
    df = pd.DataFrame({
        'Date': [],
        'Description': [],
        'Amount': [],
        'Category': []
    })
    df.to_csv(TEMP_CSV_PATH, index=False)
    return df

# Initialize the app
app = FastAPI(
    title="💼 Bank Statement Consultant API",
    description="Upload a bank statement and ask dynamic financial questions.",
    version="1.0.0"
)

# Allow frontend requests from all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory managers
memory_manager = None
statement_manager = None

# Startup event to initialize everything before the first request
@app.on_event("startup")
async def startup_event():
    global memory_manager, statement_manager, chatbot
    
    print("🚀 Starting Bank Statement Consultant API...")
    
    # Initialize Redis-backed managers
    try:
        memory_manager = ChatMemoryManager(user_id=DEFAULT_USER_ID)
        statement_manager = BankStatementMemoryManager(user_id=DEFAULT_USER_ID)
        print("✅ Memory managers initialized successfully")
    except Exception as e:
        print(f"❌ Redis connection failed: {str(e)}")
        raise RuntimeError(f"Redis connection failed. Ensure Redis is running on localhost.")
    
    # Ensure we have a file to initialize the model with
    if not os.path.exists(TEMP_CSV_PATH):
        print("📄 Creating initial empty CSV file for model initialization")
        initial_df = create_empty_dataframe()
    else:
        print("📄 Using existing CSV file for model initialization")
        initial_df = pd.read_csv(TEMP_CSV_PATH)
    
    # Initialize FinTechChatbot with the model
    try:
        print(f"🔄 Loading FinTech model from {MODEL_DIR}...")
        start_time = time.time()
        
        # Initialize chatbot globally
        chatbot = FinTechChatbot(MODEL_DIR, TEMP_CSV_PATH)
        chatbot.memory_manager = memory_manager
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds!")
    except Exception as e:
        print(f"❌ Failed to initialize model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to initialize the FinTech model: {str(e)}")


# ----------------------------
# Data Model for User Queries
# ----------------------------
class QueryModel(BaseModel):
    session_id: str
    user_query: str

# ----------------------------
# Upload Bank Statement
# ----------------------------
@app.post("/upload-statement")
async def upload_statement(session_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        
        # Save the CSV to the temporary file path
        df.to_csv(TEMP_CSV_PATH, index=False)
        print(f"📄 Saved uploaded statement to {TEMP_CSV_PATH}")
        
        # Convert DataFrame to JSON string for BankStatementMemoryManager
        statement_json = df.to_json(orient="records")
        statement_manager.save_statement_memory(statement_json)
        
        # Update the chatbot's dataframe with the new statement
        chatbot.transactions_df = df
        print("📊 Updated chatbot with new transaction data")
            
        return {"message": "✅ Bank statement uploaded and saved successfully."}
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Failed to process CSV file: {e}")

# ----------------------------
# Ask Question Based on Bank Statement
# ----------------------------
@app.post("/chat")
async def chat(data: QueryModel):
    user_query = data.user_query
    session_id = data.session_id

    print(f"💬 Received query: '{user_query}' for session: {session_id}")

    # Get stored bank statement
    statement_json = statement_manager.load_statement_memory()
    if not statement_json:
        raise HTTPException(status_code=400, detail="⚠ No bank statement found. Please upload one first.")
    
    # Convert JSON string back to DataFrame
    try:
        statement_df = pd.read_json(statement_json)
        print(f"📊 Loaded statement with {len(statement_df)} transactions")
    except Exception as e:
        print(f"❌ Statement parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"❌ Error parsing bank statement data: {e}")

    # Process the query using the pre-loaded model
    try:
        # Set the current session and data
        chatbot.session_id = session_id
        chatbot.transactions_df = statement_df
        
        # Process the query
        print(f"🧠 Processing query with model...")
        answer = chatbot.process_query(user_query)
        print(f"✅ Generated response successfully")
        
        # Update memory with new conversation
        recent_history = memory_manager.load_memory()
        recent_history.append({"role": "user", "content": user_query})
        recent_history.append({"role": "assistant", "content": answer})
        memory_manager.save_memory(recent_history)
        
        return JSONResponse(content={"response": answer})
    except Exception as e:
        print(f"❌ Query processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Error generating response: {e}")
    
# ----------------------------
# Generate Session ID (Optional)
# ----------------------------
@app.get("/session")
def generate_session():
    session_id = str(uuid.uuid4())
    print(f"🔑 Generated new session ID: {session_id}")
    return {"session_id": session_id}

# ----------------------------
# View Current Statement (Optional)
# ----------------------------
@app.get("/statement")
def get_current_statement(session_id: str):
    try:
        statement_json = statement_manager.load_statement_memory()
        if not statement_json:
            raise HTTPException(status_code=404, detail="No statement found for this session.")
        statement_df = pd.read_json(statement_json)
        print(f"📋 Retrieved statement with {len(statement_df)} rows for session: {session_id}")
        return statement_df.to_dict(orient="records")
    except Exception as e:
        print(f"❌ Error retrieving statement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving statement: {e}")
    
# cd Redis
# .\redis-server.exe
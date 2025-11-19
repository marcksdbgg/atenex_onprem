import sys
import os

# Add the project root to the python path
sys.path.append(os.getcwd())

try:
    print("Importing app.application.use_cases.ask_query_use_case...")
    from app.application.use_cases.ask_query_use_case import AskQueryUseCase
    print("AskQueryUseCase imported successfully.")
except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)

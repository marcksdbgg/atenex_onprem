import sys
import os

# Add the project root to the python path
sys.path.append(os.getcwd())

try:
    print("Importing app.application.use_cases.ask_query.config_types...")
    from app.application.use_cases.ask_query.config_types import PromptBudgetConfig
    print("config_types imported successfully.")
except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)

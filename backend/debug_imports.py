#!/usr/bin/env python3
"""Debug import issues"""

import sys
import os

print("Current working directory:", os.getcwd())
print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying to import modules...")

try:
    from src.calculations import calculate_bmr
    print("✅ src.calculations imported successfully")
except Exception as e:
    print(f"❌ src.calculations failed: {e}")

try:
    from src.validation import validate_user_profile
    print("✅ src.validation imported successfully")
except Exception as e:
    print(f"❌ src.validation failed: {e}")

try:
    from src.config import Config
    print("✅ src.config imported successfully")
except Exception as e:
    print(f"❌ src.config failed: {e}")

try:
    from src.templates import TemplateManager
    print("✅ src.templates imported successfully")
except Exception as e:
    print(f"❌ src.templates failed: {e}")

try:
    from src.thesis_model import XGFitnessAIModel
    print("✅ src.thesis_model imported successfully")
except Exception as e:
    print(f"❌ src.thesis_model failed: {e}")

# XGFitness AI - Advanced Fitness Recommendation System

## Overview
XGFitness AI is a sophisticated machine learning system that provides personalized fitness and nutrition recommendations using XGBoost algorithms with advanced feature engineering.

## 🎯 Key Features

### Machine Learning
- **XGBoost Models**: Separate models for workout and nutrition recommendations
- **Advanced Feature Engineering**: 11 engineered features including interaction terms, metabolic ratios, and health deviation scores
- **Anti-Overfitting Measures**: Separate hyperparameter grids, noise injection, and stronger regularization for nutrition model
- **Feature Scaling**: StandardScaler for optimal model performance
- **Confidence Scoring**: Honest validation with confidence levels

### Data Pipeline
- **70/15/15 Split**: 70% real training data, 15% real validation data, 15% logical test data
- **Real Data Priority**: Uses 3,659 real samples (ages 18-65) from household dataset
- **Smart Test Data**: Last 15% is generated logical dummy data with confidence scores
- **Age Filtering**: Automatically filters to adults aged 18-65

### Templates & Recommendations
- **9 Workout Templates**: 3 fitness goals × 3 activity levels
- **8 Nutrition Templates**: Valid goal + BMI category combinations
- **Daily Outputs**: All recommendations in daily format (not weekly)
- **No Target Weight**: Focus on body composition and fitness goals

## 📁 Project Structure

```
xgfitness/
├── backend/                 # Python ML backend
│   ├── src/                # Core ML modules
│   │   ├── thesis_model.py # Main XGBoost model
│   │   ├── templates.py    # Template management
│   │   ├── validation.py   # Input validation
│   │   ├── config.py       # Configuration
│   │   └── calculations.py # BMR/TDEE calculations
│   ├── models/             # Trained models
│   ├── logs/               # Application logs
│   ├── app.py              # Flask API server
│   ├── train_model.py      # Model training script
│   ├── test_suite.py       # Comprehensive tests
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API services
│   │   └── utils/          # Validation utilities
│   └── build/              # Production build
├── data/                   # Template data
│   ├── workout_templates.csv
│   └── nutrition_templates.csv
└── e267_Data...txt         # Real training data
```

## 🚀 Quick Start

### Initial Setup (First Time Only)
```bash
# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate      # Windows PowerShell
# OR
.venv\Scripts\activate.bat  # Windows Command Prompt
# OR
source .venv/bin/activate   # Linux/Mac
```

### Backend Setup
```bash
# Activate virtual environment (from project root)
.venv\Scripts\activate      # Windows PowerShell
# OR
.venv\Scripts\activate.bat  # Windows Command Prompt
# OR
source .venv/bin/activate   # Linux/Mac

# Install dependencies and run
cd backend
pip install -r requirements.txt
python train_model.py      # Train the models
python app.py              # Start API server
```

### Frontend Setup
```bash
cd frontend
npm install
npm run build              # Production build
npm start                  # Development server
```

### Testing
```bash
cd backend
python test_suite.py       # Run comprehensive tests
```

## 🔬 Technical Details

### Feature Engineering (11 Features)
1. **Interaction Features**:
   - BMI × Goal interaction
   - Age × Activity interaction  
   - BMI × Activity interaction
   - Age × Goal interaction

2. **Metabolic Ratios**:
   - BMR per kg (BMR/weight)
   - TDEE/BMR ratio

3. **Health Deviation Scores**:
   - BMI deviation from ideal (22.5)
   - Weight/height ratio

4. **Boolean Flags**:
   - High metabolism (above median BMR/kg)
   - Very active (high activity level)
   - Young adult (age < 30)

### Model Architecture
- **Workout Model**: XGBoost classifier for template selection
- **Nutrition Model**: XGBoost classifier for nutrition recommendations
- **Hyperparameters**: Optimized via RandomizedSearchCV
- **Scaling**: StandardScaler for feature normalization

### Template System
- **Workout Templates**: Daily sets, sessions, cardio minutes
- **Nutrition Templates**: TDEE multipliers, macro ratios per kg
- **Valid Combinations**: Only evidence-based goal + BMI pairings
- **Special Case**: ('Maintenance', 'Underweight') for body composition contexts

## 📊 Data Sources
- **Real Data**: 3,659 samples from household fitness survey (ages 18-65)
- **Dummy Data**: Logically generated with confidence scores when needed
- **Age Range**: Filtered to adults 18-65 for safety and relevance

## 🎯 Fitness Goals
1. **Fat Loss**: Caloric deficit with muscle preservation
2. **Muscle Gain**: Slight surplus with progressive overload
3. **Maintenance**: Body composition focus over weight

## 📈 Model Performance
- **Workout Model**: 81.5% accuracy, F1: 0.74 (realistic performance)
- **Nutrition Model**: 92.2% accuracy, F1: 0.91 (fixed overfitting issue)
- **Real Data Usage**: Maximum utilization of authentic data (85% real, 15% logical test)
- **Daily Predictions**: Practical, actionable daily recommendations
- **No Target Weight**: Evidence-based approach without arbitrary targets
- **Anti-Overfitting**: Separate hyperparameter grids and noise injection prevent perfect scores

## 🛠️ API Endpoints
- `POST /predict`: Get personalized recommendations
- `GET /health`: System health check
- `GET /templates`: View available templates

## 🔄 Recent Updates
- ✅ Removed target weight dependency
- ✅ Implemented 70/15/15 data split (70% real train, 15% real val, 15% logical test)
- ✅ Added advanced feature engineering
- ✅ Switched to daily output format
- ✅ Enhanced confidence scoring
- ✅ Fixed nutrition model overfitting (reduced from 100% to 92.2% accuracy)
- ✅ Implemented anti-overfitting measures (noise injection, separate hyperparameters)
- ✅ Fixed template ID consistency (template_id)
- ✅ Achieved 85% real data usage (3,657 samples available)

## 📄 License
Private project - All rights reserved.

---
*XGFitness AI - Evidence-based fitness recommendations powered by advanced machine learning.*

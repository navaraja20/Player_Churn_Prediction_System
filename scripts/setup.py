"""
Setup and Run Script
Complete setup for Player Churn Prediction System
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed")
        return False
    print(f"\n✅ {description} completed successfully")
    return True

def main():
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║   Player Churn Prediction System - Setup Script      ║
    ║                                                       ║
    ║   This script will:                                   ║
    ║   1. Generate synthetic player data                   ║
    ║   2. Run ETL pipeline                                 ║
    ║   3. Train ML models                                  ║
    ║   4. Prepare system for use                           ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    input("Press Enter to begin setup...")
    
    # Step 1: Create directories
    print("\n📁 Creating directories...")
    directories = [
        'data/raw',
        'data/processed',
        'data/staging',
        'data/predictions',
        'data/monitoring',
        'models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ {directory}")
    
    # Step 2: Generate data
    if not run_command(
        "python data/synthetic_data_generator.py",
        "Step 1/3: Generating synthetic player data"
    ):
        return
    
    # Step 3: Run ETL
    if not run_command(
        "python src/etl/etl_pipeline.py",
        "Step 2/3: Running ETL pipeline"
    ):
        return
    
    # Step 4: Train models
    if not run_command(
        "python scripts/train_models.py",
        "Step 3/3: Training ML models"
    ):
        return
    
    # Success
    print("""
    
    ╔═══════════════════════════════════════════════════════╗
    ║              🎉 SETUP COMPLETE! 🎉                    ║
    ╚═══════════════════════════════════════════════════════╝
    
    Your Player Churn Prediction System is ready to use!
    
    📊 Data Generated:
       - 100,000 players
       - 6 months of activity data
       - 600,000+ records
    
    🤖 Models Trained:
       - XGBoost
       - Random Forest
       - LightGBM
       - Ensemble
    
    🚀 Next Steps:
    
    1. Start the API server:
       uvicorn src.api.main:app --host 0.0.0.0 --port 8000
       
       Then visit: http://localhost:8000/docs
    
    2. Launch the dashboard:
       streamlit run streamlit/dashboard.py
       
       Then visit: http://localhost:8501
    
    3. Or use Docker:
       docker-compose up -d
    
    📚 Documentation:
       - Model Card: docs/MODEL_CARD.md
       - Technical Docs: docs/TECHNICAL_DOCUMENTATION.md
       - README: README.md
    
    💡 Quick Test:
       Check out notebooks/01_EDA.ipynb for exploration
    
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        sys.exit(1)

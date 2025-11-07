@echo off
echo =========================================
echo ⚖️  Starting LexVault AI Pipeline + Dashboard
echo =========================================

REM Activate virtual environment
call .venv\Scripts\activate

REM Run full pipeline
echo Running main pipeline...
python src\main.py

REM Launch Streamlit dashboard
echo Launching Streamlit Dashboard...
streamlit run app.py

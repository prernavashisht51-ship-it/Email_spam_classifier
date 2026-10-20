@echo off
cd /d "%~dp0"
title Email Spam Classifier
echo Starting Email Spam Classifier...
python -m streamlit run app.py
pause

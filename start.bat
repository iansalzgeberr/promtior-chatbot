@echo off
echo Starting Promtior RAG Chatbot...
echo.

REM Activar el entorno virtual
call venv\Scripts\activate.bat

REM Iniciar el backend en una nueva ventana
echo [1/2] Starting backend (LangServe) on http://localhost:8000 ...
start "Promtior Backend" cmd /k "set PYTHONIOENCODING=utf-8 && python -m app.server"

REM Esperar a que el backend arranque
timeout /t 20 /nobreak > nul

REM Iniciar el frontend en una nueva ventana
echo [2/2] Starting frontend (Streamlit) on http://localhost:8501 ...
start "Promtior Frontend" cmd /k "streamlit run frontend/streamlit_app.py"

echo.
echo Both services started!
echo   Backend  : http://localhost:8000
echo   Playground: http://localhost:8000/chat/playground
echo   Frontend : http://localhost:8501
echo.
pause

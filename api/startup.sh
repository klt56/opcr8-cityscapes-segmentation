#!/bin/bash
cd "$(dirname "$0")"

# Lancer uvicorn directement via le python du venv antenv
# (python3 résout automatiquement le bon environnement via PYTHONPATH d'Oryx)
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600

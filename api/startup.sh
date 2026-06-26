#!/bin/bash
# Script de démarrage pour Azure App Service

# Se placer dans le dossier du script (api/)
cd "$(dirname "$0")"

# Trouver et activer le venv antenv créé par Oryx
# (Oryx peut le placer dans /tmp ou dans wwwroot selon le mode de build)
VENV_PATH=$(find /tmp /home/site/wwwroot -name "activate" -path "*antenv*" 2>/dev/null | head -1)
if [ -n "$VENV_PATH" ]; then
    source "$VENV_PATH"
fi

# Lancement de l'API FastAPI avec gunicorn + uvicorn workers
gunicorn -w 1 -k uvicorn.workers.UvicornWorker \
    --bind=0.0.0.0:8000 \
    --timeout=600 \
    main:app

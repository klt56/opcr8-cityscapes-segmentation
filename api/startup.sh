#!/bin/bash
# Script de démarrage pour Azure App Service
# Lance uvicorn avec gunicorn pour gérer les requêtes en production

# Se placer dans le dossier où se trouve main.py
# (chemin relatif car Oryx extrait l'archive dans un dossier temporaire)
cd "$(dirname "$0")"

# Lancement de l'API FastAPI avec gunicorn + uvicorn workers
# - workers=1 : un seul worker (suffisant vu la taille du modèle ~95Mo)
# - timeout=600 : longue durée pour le chargement initial du modèle
# - bind 0.0.0.0:8000 : Azure expose le port 8000 par défaut
gunicorn -w 1 -k uvicorn.workers.UvicornWorker \
    --bind=0.0.0.0:8000 \
    --timeout=600 \
    main:app

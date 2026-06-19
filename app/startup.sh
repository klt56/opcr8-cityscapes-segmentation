#!/bin/bash
# Script de démarrage pour Azure App Service - Application Streamlit

cd /home/site/wwwroot/app

# Lancement de Streamlit
# - Azure expose automatiquement le port 8000
# - --server.address=0.0.0.0 pour accepter les connexions externes
# - --server.headless=true pour ne pas tenter d'ouvrir un navigateur
# - --server.enableCORS=false pour autoriser les appels depuis n'importe où
streamlit run streamlit_app.py \
    --server.port=8000 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false

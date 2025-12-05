# Multi-stage Dockerfile for Vector Precognition Deployment Suite
# Optimized for fast builds using layer caching

# ==============================================================================
# Stage 1: Base Image - Common dependencies
# ==============================================================================
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# ==============================================================================
# Stage 2: Dependencies - Install Python packages
# ==============================================================================
FROM base as dependencies

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 3: Application - Copy application code
# ==============================================================================
FROM dependencies as application

# Switch to non-root user
USER appuser

# Copy shared modules
COPY --chown=appuser:appuser shared/ ./shared/

# Copy models directory
COPY --chown=appuser:appuser models/ ./models/

# Copy environment configuration
COPY --chown=appuser:appuser .env.example ./.env.example

# Copy all application directories (keeping COPY as last step for efficiency)
COPY --chown=appuser:appuser app1_guardrail_erosion/ ./app1_guardrail_erosion/
COPY --chown=appuser:appuser app2_rho_calculator/ ./app2_rho_calculator/
COPY --chown=appuser:appuser app3_phi_evaluator/ ./app3_phi_evaluator/
COPY --chown=appuser:appuser app4_unified_dashboard/ ./app4_unified_dashboard/

# Expose ports for all 4 apps
EXPOSE 8501 8502 8503 8504

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ==============================================================================
# Stage 4: App-specific images
# ==============================================================================

# --- App 1: Guardrail Erosion Analyzer ---
FROM application as app1
WORKDIR /app/app1_guardrail_erosion
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# --- App 2: RHO Calculator ---
FROM application as app2
WORKDIR /app/app2_rho_calculator
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0", "--server.headless=true"]

# --- App 3: PHI Evaluator ---
FROM application as app3
WORKDIR /app/app3_phi_evaluator
CMD ["streamlit", "run", "app.py", "--server.port=8503", "--server.address=0.0.0.0", "--server.headless=true"]

# --- App 4: Unified Dashboard ---
FROM application as app4
WORKDIR /app/app4_unified_dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8504", "--server.address=0.0.0.0", "--server.headless=true"]

# ==============================================================================
# Default target: All-in-one image (runs App 4 by default)
# ==============================================================================
FROM application as default
WORKDIR /app
# Create entrypoint script to allow running any app
RUN echo '#!/bin/bash\n\
case "$APP" in\n\
  app1) cd app1_guardrail_erosion && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true ;;\n\
  app2) cd app2_rho_calculator && streamlit run app.py --server.port=8502 --server.address=0.0.0.0 --server.headless=true ;;\n\
  app3) cd app3_phi_evaluator && streamlit run app.py --server.port=8503 --server.address=0.0.0.0 --server.headless=true ;;\n\
  app4) cd app4_unified_dashboard && streamlit run app.py --server.port=8504 --server.address=0.0.0.0 --server.headless=true ;;\n\
  *) echo "Usage: docker run -e APP=app1|app2|app3|app4" && exit 1 ;;\n\
esac\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENV APP=app4
ENTRYPOINT ["/app/entrypoint.sh"]

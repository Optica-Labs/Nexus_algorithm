# 📊 Visualization Output Guide

## Directory Structure

Plots are organized by mode:

```
output/
├── text/                    # Text mode (AWS Bedrock + PCA)
│   ├── conversation_dynamics_<timestamp>.png
│   └── conversation_summary_<timestamp>.png
│
├── manual/                  # Manual mode (hardcoded 2D vectors)
│   ├── conversation_dynamics_<timestamp>.png
│   └── conversation_summary_<timestamp>.png
│
└── visuals/                 # Legacy plots (before mode separation)
```

## Plot Types

### 1. Conversation Dynamics (4-Panel Plot)

Shows complete risk trajectory for a single conversation:

- **Panel 1: Risk Severity R(N)** - Distance from safe-harbor (0-2)
- **Panel 2: Risk Rate v(N)** - First derivative (velocity)
- **Panel 3: Guardrail Erosion a(N)** - Second derivative (acceleration)
- **Panel 4: Likelihood L(N)** - Breach probability (0-1)

### 2. Conversation Summary (Scatter Plot)

Compares multiple conversations:

- **X-axis**: Peak Risk Severity
- **Y-axis**: Peak Risk Likelihood
- **Red shading**: Danger zone (high risk)

## Usage

### Manual Mode
```bash
python src/vector_precognition_demo.py --mode manual
```
Saves to `output/manual/`

### Text Mode
```bash
python src/vector_precognition_demo.py --mode text
```
Saves to `output/text/`

## Features

✅ High-resolution (150 DPI)  
✅ Timestamped filenames  
✅ Mode-specific directories  
✅ Console feedback  
✅ Works headless

For detailed analysis, see `README.md`.

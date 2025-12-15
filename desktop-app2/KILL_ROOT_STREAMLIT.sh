#!/bin/bash

echo "====================================="
echo "Killing root Streamlit process"
echo "====================================="
echo ""
echo "This will kill the Streamlit process running as root on port 8501"
echo "You will be prompted for your sudo password."
echo ""

sudo kill -9 2530

echo ""
echo "Process killed! Now you can run ./START_FINAL.sh"

#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y libgl1-mesa-glx

# Run your Streamlit app
streamlit run app.py --server.port 8000 --server.address 0.0.0.0

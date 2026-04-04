FROM python:3.10-slim

# Added requests so Streamlit can talk to the FastAPI backend
RUN pip install --no-cache-dir streamlit requests

WORKDIR /app

EXPOSE 8501

# Cleaned up the redundant flags
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
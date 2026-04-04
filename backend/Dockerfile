FROM continuumio/miniconda3

# Create conda environment with Python 3.10 and CadQuery from conda-forge
RUN conda create -n cadenv python=3.10 -y && \
    conda install -n cadenv -c conda-forge cadquery -y && \
    conda clean -afy

# Install Python dependencies inside the conda environment
RUN /opt/conda/envs/cadenv/bin/pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    google-generativeai

WORKDIR /app

EXPOSE 8000

# Activate conda env and run uvicorn
CMD ["conda", "run", "--no-capture-output", "-n", "cadenv", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

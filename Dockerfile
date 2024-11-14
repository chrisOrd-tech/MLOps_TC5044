FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the model and server code
COPY models/model_log_reg.pkl /app/
COPY src/api.py /app/

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn pydantic

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
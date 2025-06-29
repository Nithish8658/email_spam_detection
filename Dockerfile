# Use a lightweight Python image
FROM python:3.9

# Set working directory
WORKDIR C:/Users/nithish/Desktop/email_spam_detection/backend/app/app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir \
    mlflow \
    psycopg2-binary \
    azure-storage-blob \
    azure-identity

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:5432/$POSTGRES_DB?sslmode=require \
    --artifacts-destination wasbs://$BLOB_CONTAINER@$STORAGE_ACCOUNT.blob.core.windows.net/ \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 1
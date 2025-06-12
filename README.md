# Workflow MLflow CI/CD for Wine Quality

## Structure

- MLProject: MLflow Project code (modelling.py, conda.yaml, dataset, MLproject file)
- Dockerfile: Build image for MLflow run
- .github/workflows/mlflow_ci.yml: GitHub Actions workflow

## DockerHub Images

- [https://hub.docker.com/r/<your-docker-username>/wine-mlflow](https://hub.docker.com/r/<your-docker-username>/wine-mlflow)

## How it works

- On push/PR: Docker image built & pushed
- MLflow Project run in container
- Model/artefak diupload ke GitHub Actions (skilled/advanced)
- (Advanced: add Google Drive or LFS upload if needed)

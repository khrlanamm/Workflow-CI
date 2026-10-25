# Workflow CI: crop recommendation model

MLflow project that trains a crop recommendation model in GitHub Actions on every push to `main`, then packages the model as a Docker image and pushes it to Docker Hub as [`khrlanamm/crop-recommendation-model`](https://hub.docker.com/r/khrlanamm/crop-recommendation-model).

Built for the Building Machine Learning Systems class at Dicoding. The data comes from [Eksperimen_SML_Khoirul_Anam](https://github.com/khrlanamm/Eksperimen_SML_Khoirul_Anam), which explores and preprocesses the Crop Recommendation dataset.

## Pipeline

1. Install MLflow, scikit-learn, pandas, and NumPy on an Ubuntu runner with Python 3.12.
2. Run the MLflow project in `MLProject/`: train a random forest (100 trees) on the preprocessed data with an 80/20 split, and log accuracy and the model.
3. Upload the `mlruns/` folder as a workflow artifact.
4. Build a serving image with `mlflow models build-docker` and push it to Docker Hub.

## Files

```
.github/workflows/mlflow_ci.yml   the CI workflow
MLProject/MLproject               MLflow entry point
MLProject/conda.yaml              environment
MLProject/modelling.py            training script
MLProject/dataset_preprocessing/  scaled features and encoded labels
```

## Run locally

```sh
pip install mlflow scikit-learn pandas numpy
mlflow run MLProject --env-manager=local
```

The workflow needs the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets for the Docker steps.

## Use the image

```sh
docker run -p 5001:8080 khrlanamm/crop-recommendation-model:latest
```

The container serves the model with MLflow's scoring server at `/invocations`.

## License

Code is released under the [MIT License](LICENSE). The data in `MLProject/dataset_preprocessing/` is derived from the public Crop Recommendation dataset and keeps its original terms.

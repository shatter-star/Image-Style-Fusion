from fastapi import FastAPI, File, UploadFile
import tempfile
import shutil
import mlflow.pytorch
from NST import neural_style_transfer

app = FastAPI()

mlflow.set_tracking_uri("https://dagshub.com/shatter-star/Image-Style-Fusion.mlflow")
mlflow.set_experiment("Neural-Style-Transfer")

def run_neural_style_transfer(config):
    # Perform neural style transfer
    results_path = neural_style_transfer(config)
    return results_path

@app.post("/predict/")
async def predict_image(content_image: UploadFile = File(...), style_image: UploadFile = File(...)):
    # Create temporary directories for content and style images
    with tempfile.TemporaryDirectory() as content_temp_dir, tempfile.TemporaryDirectory() as style_temp_dir:
        # Save the content and style images to temporary directories
        content_image_path = f"{content_temp_dir}/content_image.jpg"
        style_image_path = f"{style_temp_dir}/style_image.jpg"
        with open(content_image_path, "wb") as content_file:
            shutil.copyfileobj(content_image.file, content_file)
        with open(style_image_path, "wb") as style_file:
            shutil.copyfileobj(style_image.file, style_file)

        # Load the model using MLflow and retrieve the run ID
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[client.get_experiment_by_name("Neural-Style-Transfer").experiment_id])
        run_id = runs[0].info.run_id

        # Prepare configuration
        config = {
            "content_weight": 100000.0,
            "style_weight": 30000.0,
            "tv_weight": 1.0,
            "content_images_dir": content_temp_dir,
            "content_img_name": "content_image.jpg",
            "style_images_dir": style_temp_dir,
            "style_img_name": "style_image.jpg",
            "output_img_dir": "data/output-images",  # Specify a fixed directory here
            "height": 400,
            "img_format": (4, '.jpg'),
            "run_id": run_id
        }

        # Perform neural style transfer
        results_path = run_neural_style_transfer(config)

    return {"message": "Neural Style Transfer completed successfully.", "results_path": results_path}

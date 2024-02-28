import mlflow
import mlflow.pytorch
import os
from NST import neural_style_transfer
from models.definitions.vgg19 import Vgg19

# Configuration dictionary
config = {
    "content_weight": 100000.0,
    "style_weight": 30000.0,
    "tv_weight": 1.0,
    "content_images_dir": "data/content-images",
    "content_img_name": "a1.jpg",
    "style_images_dir": "data/style-images",
    "style_img_name": "b1.jpg",
    "output_img_dir": "data/output-images",
    "height": 400,
    "img_format": (4, '.jpg')
}

def run_neural_style_transfer(config):
    # Set tracking URI
    mlflow.set_tracking_uri("/home/siddharth/Desktop/Neural-Style-Transfer/mlruns")
    # Set the experiment name
    mlflow.create_experiment("NeuralStyleTransferExperiment-6")
    mlflow.set_experiment("NeuralStyleTransferExperiment-6")

    # Instantiate Vgg19 model
    vgg_model = Vgg19()

    with mlflow.start_run():
        # Log parameters
        for key, value in config.items():
            mlflow.log_param(key, value)

        # Run neural style transfer and get dump path
        dump_path = neural_style_transfer(config)

        # Log output image
        output_image_path = os.path.join(dump_path, config['content_img_name'].split('.')[0] + '_b1.jpg')
        mlflow.log_artifact(output_image_path, artifact_path="output_images")

        # Log the whole model
        mlflow.pytorch.log_model(vgg_model, "model")
        mlflow.pytorch.log_model(vgg_model, "model", registered_model_name="VGG19Model")

if __name__ == "__main__":
    # Create the output image directory if it doesn't exist
    os.makedirs(config['output_img_dir'], exist_ok=True)
    
    run_neural_style_transfer(config)

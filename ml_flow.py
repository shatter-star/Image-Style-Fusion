import mlflow
import mlflow.pytorch
import os
import argparse
import matplotlib.pyplot as plt
from NST import neural_style_transfer
from models.definitions.vgg19 import Vgg19

# Configuration dictionary
DEFAULT_CONFIG = {
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
    # Instantiate Vgg19 model
    vgg_model = Vgg19()

    loss_history = []
    # Set the active experiment
    experiment_name = "Neural-Style-Transfer"
    mlflow.set_experiment(experiment_name)
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        for key, value in config.items():
            mlflow.log_param(key, value)

        # Run neural style transfer and get dump path
        dump_path, loss_history = neural_style_transfer(config)

        # Log output image
        output_image_path = os.path.join(dump_path, config['content_img_name'].split('.')[0] + '_b1.jpg')
        mlflow.log_artifact(output_image_path, artifact_path="output_images")

        # Log metrics
        log_interval = 5  # Adjust as needed
        for i, (total_loss, content_loss, style_loss, tv_loss) in enumerate(loss_history):
            if i % log_interval == 0:
                mlflow.log_metric(f'total_loss_{i}', total_loss)
                mlflow.log_metric(f'content_loss_{i}', content_loss)
                mlflow.log_metric(f'style_loss_{i}', style_loss)
                mlflow.log_metric(f'tv_loss_{i}', tv_loss)


        # Log the whole model
        mlflow.pytorch.log_model(vgg_model, "model")
        mlflow.pytorch.log_model(vgg_model, "model", registered_model_name="VGG19Model")

        # Extract loss values from loss_history
        total_loss_values = [loss[0] for loss in loss_history]
        style_loss_values = [loss[2] for loss in loss_history]  # Style loss is at index 2
        content_loss_values = [loss[1] for loss in loss_history]  # Content loss is at index 1
        tv_loss_values = [loss[3] for loss in loss_history]

        # Plotting the loss curve
        plt.figure(figsize=(10, 6))  # Adjust figure size if needed
        plt.plot(total_loss_values, label='Total Loss')
        plt.plot(style_loss_values, label='Style Loss')
        plt.plot(content_loss_values, label='Content Loss')
        plt.plot(tv_loss_values, label='TV Loss')

        # Add labels and title to the plot
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()

        # Save and log the plot
        plt.yscale('log')
        plt.savefig('loss_curve.png')
        mlflow.log_artifact('loss_curve.png')



def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content_weight', type=float, default=DEFAULT_CONFIG['content_weight'],
                        help='Weight for content loss')
    parser.add_argument('--style_weight', type=float, default=DEFAULT_CONFIG['style_weight'],
                        help='Weight for style loss')
    parser.add_argument('--tv_weight', type=float, default=DEFAULT_CONFIG['tv_weight'],
                        help='Weight for total variation loss')
    parser.add_argument('--content_images_dir', type=str, default=DEFAULT_CONFIG['content_images_dir'],
                        help='Directory containing content images')
    parser.add_argument('--content_img_name', type=str, default=DEFAULT_CONFIG['content_img_name'],
                        help='Name of content image')
    parser.add_argument('--style_images_dir', type=str, default=DEFAULT_CONFIG['style_images_dir'],
                        help='Directory containing style images')
    parser.add_argument('--style_img_name', type=str, default=DEFAULT_CONFIG['style_img_name'],
                        help='Name of style image')
    parser.add_argument('--output_img_dir', type=str, default=DEFAULT_CONFIG['output_img_dir'],
                        help='Directory to save output images')
    parser.add_argument('--height', type=int, default=DEFAULT_CONFIG['height'],
                        help='Height of output image')
    parser.add_argument('--img_format', nargs=2, default=DEFAULT_CONFIG['img_format'], metavar=('size', 'extension'),
                        help='Image format size and extension')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    config = vars(args)
    os.makedirs(config['output_img_dir'], exist_ok=True)
    
    # Check if the experiment exists, if not, create it
    experiment_name = "Neural-Style-Transfer"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    
    # Run neural style transfer
    run_neural_style_transfer(config)

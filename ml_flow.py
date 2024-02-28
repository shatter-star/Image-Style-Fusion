# neural_style_transfer.py

import mlflow
import mlflow.pytorch
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import LBFGS
import os
from models.definitions.vgg19 import Vgg19
from NST import load_image, prepare_img, save_image, generate_out_img_name, save_and_maybe_display, gram_matrix, build_loss, prepare_model

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

def neural_style_transfer(config):
    '''
    The main Neural Style Transfer method.
    '''
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = prepare_img(content_img_path, config['height'], device)
    style_img = prepare_img(style_img_path, config['height'], device)
    
    init_img = content_img
    
    optimizing_img = Variable(init_img, requires_grad=True)
    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_model(device)
    print(f'Using VGG19 in the optimization procedure.')
    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)
    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]
    num_of_iterations = 20
    
    optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
    cnt = 0
    mlflow.set_tracking_uri("/home/siddharth/Desktop/Neural-Style-Transfer/mlruns")
    # Set the experiment name
    mlflow.set_experiment("NeuralStyleTransferExperiment")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("content_weight", config["content_weight"])
        mlflow.log_param("style_weight", config["style_weight"])
        mlflow.log_param("tv_weight", config["tv_weight"])

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations)
            cnt += 1
            return total_loss
        optimizer.step(closure)

        # Log the model
        mlflow.pytorch.log_model(neural_net, "model")

        # Register the model
        mlflow.pytorch.log_model(neural_net, "model", registered_model_name="VGG19Model")

    return dump_path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Assuming you are using YOLO from the ultralytics library

class YOLOActivationHelper:
    def __init__(self, model_path):
        """
        Initialize the YOLOActivationHelper with a YOLO model.

        :param model_path: Path to the YOLO model weights file.
        """
        self.model = YOLO(model_path)

    def get_target_layer_name(self, index=-3):
        """
        Get the name of a target layer in the YOLO model.

        :param index: Index of the desired layer (default is -3 for the third last layer).
        :return: Name of the target layer.
        """
        layer_names = [name for name, _ in self.model.model.named_modules()]
        if len(layer_names) > abs(index):
            return layer_names[index]
        raise ValueError("Invalid layer index specified.")

    def get_activation_heatmap(self, image_path, layer_name):
        """
        Generate an activation heatmap for a specific layer.

        :param image_path: Path to the input image.
        :param layer_name: Name of the layer to visualize.
        :return: Tuple of (original_image, heatmap, overlay).
        """
        # Load image and preprocess
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (640, 640))  # Resize to match YOLO input size
        input_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # Forward pass and hook to get the activation
        activations = []

        def hook_fn(module, input, output):
            activations.append(output)

        layer = dict(self.model.model.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.model(input_tensor)

        hook.remove()

        # Extract the activations
        activation_map = activations[0][0].mean(dim=0).cpu().numpy()  # Average over channels
        activation_map = cv2.resize(activation_map, (image.shape[1], image.shape[0]))  # Resize to original image size

        # Normalize activation map
        activation_map_normalized = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * activation_map_normalized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay heatmap on the original image
        overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

        return image_rgb, heatmap, overlay

    @staticmethod
    def plot_results(original, heatmap, overlay):
        """
        Plot the original image, heatmap, and overlay.

        :param original: Original image.
        :param heatmap: Heatmap image.
        :param overlay: Overlay of heatmap on original image.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(heatmap)
        axes[1].set_title("Activation Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Heatmap Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    model_path = "yolo11m.pt"
    image_path = "/content/dc_2.jpg"

    # Initialize helper
    helper = YOLOActivationHelper(model_path)

    # Get target layer name
    target_layer_name = helper.get_target_layer_name()

    # Generate heatmaps
    original, heatmap, overlay = helper.get_activation_heatmap(image_path, target_layer_name)

    # Plot results
    helper.plot_results(original, heatmap, overlay)

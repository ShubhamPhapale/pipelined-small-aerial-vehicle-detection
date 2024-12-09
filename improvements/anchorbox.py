##Anchorbox size

## change should reflect in the model.yaml file
## print ke bajay reflect kro

import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

class IOUKMeans:
    def __init__(self, n_clusters=9, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def calculate_iou(self, box, centroid):
        """Calculate IoU between a box and a centroid"""
        w1, h1 = box
        w2, h2 = centroid
        intersection = min(w1, w2) * min(h1, h2)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0

    def iou_distance(self, box, centroid):
        """Calculate distance using d = 1 - IoU"""
        return 1 - self.calculate_iou(box, centroid)

    def fit(self, boxes):
        """Perform k-means clustering using IoU-based distance"""
        num_boxes = len(boxes)

        # Initialize centroids randomly from the data
        centroid_indices = random.sample(range(num_boxes), self.n_clusters)
        self.centroids = boxes[centroid_indices].copy()

        prev_assignments = np.ones(num_boxes) * -1

        for iteration in range(self.max_iter):
            # Calculate distances between boxes and centroids
            distances = np.zeros((num_boxes, self.n_clusters))
            for i in range(num_boxes):
                for j in range(self.n_clusters):
                    distances[i][j] = self.iou_distance(boxes[i], self.centroids[j])

            # Assign boxes to nearest centroid
            assignments = np.argmin(distances, axis=1)

            # Check for convergence
            if (assignments == prev_assignments).all():
                break

            # Update centroids
            for i in range(self.n_clusters):
                cluster_boxes = boxes[assignments == i]
                if len(cluster_boxes) > 0:
                    self.centroids[i] = np.mean(cluster_boxes, axis=0)

            prev_assignments = assignments.copy()

        # Sort centroids by area
        areas = self.centroids[:, 0] * self.centroids[:, 1]
        sort_indices = np.argsort(areas)
        self.centroids = self.centroids[sort_indices]

        return self.centroids

class YOLOAnchorOptimizer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.bbox_data = None

    def load_labels(self, split='train'):
        labels_path = self.dataset_path / split / 'labels'
        all_boxes = []

        print(f"Loading {split} labels...")
        for label_file in tqdm(list(labels_path.glob('*.txt'))):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, width, height = map(float, parts)
                        all_boxes.append([width, height])

        self.bbox_data = np.array(all_boxes)
        print(f"Loaded {len(self.bbox_data)} bounding boxes")
        return self.bbox_data

    def analyze_boxes(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.bbox_data[:, 0], self.bbox_data[:, 1], alpha=0.1, s=1)
        plt.title('Width-Height Distribution')
        plt.xlabel('Normalized Width')
        plt.ylabel('Normalized Height')
        plt.grid(True)
        plt.show()

        print("\nBox Dimensions Summary (normalized):")
        print(f"Width  - Mean: {self.bbox_data[:, 0].mean():.4f}, "
              f"Std: {self.bbox_data[:, 0].std():.4f}, "
              f"Min: {self.bbox_data[:, 0].min():.4f}, "
              f"Max: {self.bbox_data[:, 0].max():.4f}")
        print(f"Height - Mean: {self.bbox_data[:, 1].mean():.4f}, "
              f"Std: {self.bbox_data[:, 1].std():.4f}, "
              f"Min: {self.bbox_data[:, 1].min():.4f}, "
              f"Max: {self.bbox_data[:, 1].max():.4f}")

    def generate_anchors(self, input_size=640):
        """Generate 9 anchor boxes using IoU k-means clustering"""
        print("Generating 9 anchor boxes using IoU-based k-means...")

        # Perform IoU-based k-means clustering
        kmeans = IOUKMeans(n_clusters=9)
        anchor_boxes = kmeans.fit(self.bbox_data)

        # Calculate average IoU
        avg_iou = 0
        for box in self.bbox_data:
            ious = [kmeans.calculate_iou(box, anchor) for anchor in anchor_boxes]
            avg_iou += max(ious)
        avg_iou /= len(self.bbox_data)

        # Convert to pixel space
        anchors_pixel = anchor_boxes * input_size

        # Print results
        print(f"\nAverage IoU with 9 anchors: {avg_iou:.4f}")

        print("\nAnchor Boxes by Scale (pixel space):")
        for i in range(3):
            start_idx = i * 3
            end_idx = start_idx + 3
            scale_anchors = anchors_pixel[start_idx:end_idx]
            print(f"Scale {i + 1} ({input_size // (8 * 2**i)}x{input_size // (8 * 2**i)}):")
            for w, h in scale_anchors:
                print(f"  ({w:3.1f}, {h:3.1f})")

        # Visualize anchors
        plt.figure(figsize=(10, 6))
        plt.scatter(self.bbox_data[:, 0], self.bbox_data[:, 1],
                   alpha=0.1, s=1, label='Ground Truth')
        plt.scatter(anchor_boxes[:, 0], anchor_boxes[:, 1],
                   c='red', s=100, marker='*', label='Anchors')
        plt.title('Anchor Boxes vs Ground Truth Boxes')
        plt.xlabel('Normalized Width')
        plt.ylabel('Normalized Height')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Format for YOLO config
        yolo_format = [f"{w:.1f},{h:.1f}" for w, h in anchors_pixel]
        print("\nYOLOv5 anchors format:")
        print(" ".join(yolo_format))

        return anchor_boxes, anchors_pixel, avg_iou

# Example usage
if __name__ == "__main__":
    dataset_path = "/content/FYP_2024/Smallmerged"
    optimizer = YOLOAnchorOptimizer(dataset_path)

    # Load data
    optimizer.load_labels(split='train')

    # Analyze current distribution
    optimizer.analyze_boxes()

    # Generate anchors
    anchor_boxes, anchors_pixel, avg_iou = optimizer.generate_anchors(input_size=640)
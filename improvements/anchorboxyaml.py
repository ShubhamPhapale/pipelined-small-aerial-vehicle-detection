import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import yaml

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
    def __init__(self, dataset_path, yaml_path):
        self.dataset_path = Path(dataset_path)
        self.yaml_path = Path(yaml_path)
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

    def generate_anchors(self, input_size=640):
        """Generate 9 anchor boxes using IoU-based k-means clustering"""
        print("Generating 9 anchor boxes using IoU-based k-means...")

        kmeans = IOUKMeans(n_clusters=9)
        anchor_boxes = kmeans.fit(self.bbox_data)

        avg_iou = 0
        for box in self.bbox_data:
            ious = [kmeans.calculate_iou(box, anchor) for anchor in anchor_boxes]
            avg_iou += max(ious)
        avg_iou /= len(self.bbox_data)

        anchors_pixel = anchor_boxes * input_size

        print(f"\nAverage IoU with 9 anchors: {avg_iou:.4f}")
        return anchors_pixel, avg_iou

    def update_yaml(self, anchors_pixel):
        """Update the YAML file with new anchor box values, distributed into 6-value lines"""
        print(f"Updating YAML file: {self.yaml_path}")
        
        # Read the original file content
        with open(self.yaml_path, 'r') as f:
            lines = f.readlines()
        
        # Find the anchors section
        anchor_start = next(i for i, line in enumerate(lines) if 'anchors:' in line)
        
        # Flatten the new_anchors list
        flat_anchors = [val for pair in anchors_pixel for val in pair] ## round(val, 2)
        
        # Distribute 18 values into 3 lines of 6 values each
        distributed_anchors = [
            flat_anchors[0:6],     # First 6 values
            flat_anchors[6:12],    # Next 6 values
            flat_anchors[12:18]    # Last 6 values
        ]
        
        # Replace anchor values while preserving comments
        new_anchor_lines = [
            f"  - [{', '.join(map(str, map(float, anchor)))}] # {comment}\n" 
            for anchor, comment in zip(
                distributed_anchors, 
                ['P3/8', 'P4/16', 'P5/32']
            )
        ]
        
        # Replace the old anchor lines with new ones
        lines[anchor_start+1:anchor_start+4] = new_anchor_lines
    
        # Write back to the file
        with open(self.yaml_path, 'w') as f:
            f.writelines(lines)
        print("YAML file updated successfully!")

# Example usage
if __name__ == "__main__":
    dataset_path = "/content/FYP_2024/Smallmerged"
    yaml_path = "/content/yolov5-model.yaml"
    optimizer = YOLOAnchorOptimizer(dataset_path, yaml_path)

    # Load dataset labels
    optimizer.load_labels(split='train')

    # Generate optimized anchors
    anchors_pixel, avg_iou = optimizer.generate_anchors(input_size=640)

    # Update anchors in the .yaml file
    optimizer.update_yaml(anchors_pixel)

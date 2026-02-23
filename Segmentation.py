from ultralytics import YOLO
import numpy as np
import cv2
import os
from scipy import interpolate
from scipy.spatial import distance as scipy_distance

model = YOLO("weights/TheFlintAI.pt")
conf = 0.36

def image_segmentation(img):
    results = model(img, conf=conf)
    results[0].show()
    return results   

def save_segmentation_image(img, output_path=None):
    results = model(img, conf=conf)
    if output_path is None:
        output_path = img.rsplit('.', 1)[0] + '_result.jpg'
    results[0].save(filename=output_path)
    return output_path   

def get_polygon(img):
    results = image_segmentation(img)[0]
    if results.masks == None:
        return {}
    polygons = {}
    length = len(results.masks.xy)
    for i in range (length):
        poly = results.masks.xy[i]  
        cls_id = int(results.boxes.cls[i])
        class_name = results.names[cls_id]
        if class_name not in polygons:
            polygons[class_name] = []
        polygons[class_name].append(poly.tolist())
    return polygons

def get_meter(img):
    polygon = get_polygon(img)
    if polygon == {}:
        return []
    else:
        return get_polygon(img).get("Meter",[])   

def get_artifact(img):
    polygon = get_polygon(img)
    if polygon == {}:
        return []
    else:
        return get_polygon(img).get("Artifact",[])
    
def get_meter_x_length(img):
    meter_polygons = get_meter(img)
    if meter_polygons == []:
        return None 
    
    x_values = []

    for polygon in meter_polygons:
        for point in polygon:
            x = point[0]
            x_values.append(x)

    x_length = max(x_values) - min(x_values)
    return x_length

def get_artifact_polygon_cm(img, meter_length_cm=5):
    meter_pixel_length = get_meter_x_length(img)
    if meter_pixel_length == None or meter_pixel_length == 0:
        return None
    
    px_cm = meter_length_cm / meter_pixel_length

    artifact_polygons = get_artifact(img)
    if artifact_polygons == []:
        return []
    
    artifact_polygons_cm = []
    for polygon in artifact_polygons:
        polygon_cm = [[x * px_cm, y * px_cm] for x, y in polygon]
        artifact_polygons_cm.append(polygon_cm)

    return artifact_polygons_cm

def resample_polygon(polygon, num_points=100):
    polygon = np.array(polygon)
    if len(polygon) < 3:
        return polygon
    
    distances = np.zeros(len(polygon))
    for i in range(1, len(polygon)):
        distances[i] = distances[i-1] + np.linalg.norm(polygon[i] - polygon[i-1])
    distances = distances / distances[-1]
    
    x_interp = interpolate.interp1d(distances, polygon[:, 0], kind='linear')
    y_interp = interpolate.interp1d(distances, polygon[:, 1], kind='linear')
    
    new_distances = np.linspace(0, 1, num_points)
    resampled = np.column_stack([x_interp(new_distances), y_interp(new_distances)])
    
    return resampled

def compute_centroid(polygon):
    polygon = np.array(polygon)
    x = polygon[:, 0]
    y = polygon[:, 1]
    return np.array([np.mean(x), np.mean(y)])

def compute_centroid_distance_signature(polygon, num_points=100):
    resampled = resample_polygon(polygon, num_points)
    centroid = compute_centroid(resampled)
    
    distances = []
    for point in resampled:
        d = np.linalg.norm(point - centroid)
        distances.append(d)
    
    return np.array(distances)

def compute_curvature(polygon, window_size=5):
    resampled = resample_polygon(polygon)
    n = len(resampled)
    curvatures = []
    
    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        
        points = resampled[start:end]
        if len(points) < 3:
            curvatures.append(0)
            continue
        
        v1 = points[0] - points[len(points)//2]
        v2 = points[-1] - points[len(points)//2]
        
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        if norm > 0:
            curvature = cross / norm
        else:
            curvature = 0
        
        curvatures.append(curvature)
    
    return np.array(curvatures)

def compute_convexity(polygon):
    polygon = np.array(polygon)
    if len(polygon) < 3:
        return 1.0
    
    try:
        hull = cv2.convexHull(polygon.astype(np.float32))
        hull_area = cv2.contourArea(hull)
        
        polygon_area = cv2.contourArea(polygon.astype(np.float32))
        
        if polygon_area == 0:
            return 1.0
        
        convexity = hull_area / polygon_area
        return convexity
    except:
        return 1.0

def compute_concave_points(polygon, threshold=0.1):
    polygon = np.array(polygon)
    n = len(polygon)
    concave_count = 0
    
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i - 1) % n]
        p2 = polygon[(i + 1) % n]
        
        v1 = p1 - p0
        v2 = p2 - p0
        
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm > 0:
            angle = np.abs(np.arcsin(cross / norm))
            if angle > np.pi / 2 + threshold:
                concave_count += 1
    
    return concave_count

def compute_protruding_points(polygon, threshold=0.1):
    polygon = np.array(polygon)
    n = len(polygon)
    protruding_count = 0
    
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i - 1) % n]
        p2 = polygon[(i + 1) % n]
        
        v1 = p1 - p0
        v2 = p2 - p0
        
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm > 0:
            angle = np.abs(np.arcsin(cross / norm))
            if angle < np.pi / 2 - threshold:
                protruding_count += 1
    
    return protruding_count

def compute_chain_code(polygon, num_directions=8):
    resampled = resample_polygon(polygon, 64)
    angles = np.arctan2(resampled[:, 1] - np.mean(resampled[:, 1]), 
                        resampled[:, 0] - np.mean(resampled[:, 0]))
    quantized = (angles + np.pi) / (2 * np.pi) * num_directions
    quantized = quantized.astype(int) % num_directions
    
    chain = np.diff(quantized)
    chain = (chain + num_directions) % num_directions
    
    return chain

def compute_fourier_descriptors(polygon, num_descriptors=10):
    resampled = resample_polygon(polygon, 64)
    centroid = compute_centroid(resampled)
    complex_coords = (resampled[:, 0] - centroid[0]) + 1j * (resampled[:, 1] - centroid[1])
    
    fourier = np.fft.fft(complex_coords)
    fourier = np.abs(fourier)
    
    normalized = fourier[1:num_descriptors+1] / fourier[0] if fourier[0] != 0 else fourier[1:num_descriptors+1]
    
    return normalized

def compute_shape_complexity(polygon):
    polygon = np.array(polygon)
    perimeter = cv2.arcLength(polygon.astype(np.float32), True)
    area = cv2.contourArea(polygon.astype(np.float32))
    
    if area == 0:
        return 0
    
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    
    return compactness

def compute_boundary_frequency(polygon, num_bins=8):
    resampled = resample_polygon(polygon)
    angles = np.arctan2(resampled[:, 1] - np.mean(resampled[:, 1]), 
                        resampled[:, 0] - np.mean(resampled[:, 0]))
    
    hist, _ = np.histogram(angles, bins=num_bins, range=(-np.pi, np.pi))
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    
    return hist

def compute_full_shape_features(polygon):
    polygon = np.array(polygon)
    if len(polygon) < 3:
        return np.zeros(60)
    
    resampled = resample_polygon(polygon, 100)
    if len(resampled) < 3:
        return np.zeros(60)
    
    features = []
    
    try:
        convexity = compute_convexity(resampled)
        features.append(convexity)
    except:
        features.append(1.0)
    
    try:
        concave_count = compute_concave_points(resampled)
        features.append(concave_count)
    except:
        features.append(0)
    
    try:
        protruding_count = compute_protruding_points(resampled)
        features.append(protruding_count)
    except:
        features.append(0)
    
    try:
        complexity = compute_shape_complexity(resampled)
        features.append(complexity)
    except:
        features.append(0)
    
    try:
        centroid_sig = compute_centroid_distance_signature(resampled, 50)
        features.extend(centroid_sig[:20])
    except:
        features.extend([0] * 20)
    
    try:
        curvature = compute_curvature(resampled)
        features.append(np.mean(np.abs(curvature)))
        features.append(np.std(curvature))
        features.append(np.max(curvature))
        features.append(np.min(curvature))
    except:
        features.extend([0, 0, 0, 0])
    
    try:
        chain = compute_chain_code(resampled)
        hist, _ = np.histogram(chain, bins=8, range=(0, 8))
        features.extend(hist / np.sum(hist) if np.sum(hist) > 0 else hist)
    except:
        features.extend([0] * 8)
    
    try:
        fourier = compute_fourier_descriptors(resampled, 10)
        features.extend(fourier)
    except:
        features.extend([0] * 10)
    
    try:
        boundary_freq = compute_boundary_frequency(resampled, 8)
        features.extend(boundary_freq)
    except:
        features.extend([0] * 8)
    
    try:
        x = resampled[:, 0]
        y = resampled[:, 1]
        features.append(np.mean(x))
        features.append(np.std(x))
        features.append(np.mean(y))
        features.append(np.std(y))
        
        width = np.max(x) - np.min(x)
        height = np.max(y) - np.min(y)
        features.append(width)
        features.append(height)
        features.append(width / height if height != 0 else 0)
    except:
        features.extend([0, 0, 0, 0, 0, 0, 0])
    
    try:
        area = cv2.contourArea(resampled.astype(np.float32))
        features.append(area)
        features.append(cv2.arcLength(resampled.astype(np.float32), True))
    except:
        features.extend([0, 0])
    
    while len(features) < 60:
        features.append(0)
    
    return np.array(features[:60])

def compare_full_shapes(poly1, poly2, img1=None, img2=None):
    f1 = compute_full_shape_features(poly1)
    f2 = compute_full_shape_features(poly2)
    
    if len(f1) != len(f2):
        min_len = min(len(f1), len(f2))
        f1 = f1[:min_len]
        f2 = f2[:min_len]
    
    f1 = np.array(f1, dtype=float)
    f2 = np.array(f2, dtype=float)
    
    max_vals = np.abs(f1) + np.abs(f2) + 1e-10
    
    f1_normalized = f1 / max_vals
    f2_normalized = f2 / max_vals
    
    euclidean_dist = np.sqrt(np.sum((f1_normalized - f2_normalized) ** 2))
    
    cosine_dist = scipy_distance.cosine(f1_normalized, f2_normalized) if np.any(f1_normalized) and np.any(f2_normalized) else 0
    
    euclidean_score = min(euclidean_dist * 10, 50)
    cosine_score = min(cosine_dist * 50, 50)
    
    combined_score = euclidean_score + cosine_score
    
    similarity_percent = 100 - min(max(combined_score, 0), 100)
    
    return similarity_percent

def polygon_features(polygon, img):
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError("Image could not be loaded")

    polygon = np.array(polygon, dtype=np.int32)

    x = polygon[:, 0]
    y = polygon[:, 1]

    width = x.max() - x.min()
    height = y.max() - y.min()
    aspect_ratio = width / height if height != 0 else 0

    area = 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )

    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    perimeter = np.sum(np.sqrt(dx**2 + dy**2))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    mean_bgr = cv2.mean(img, mask=mask)[:3] 
    mean_rgb = mean_bgr[::-1]          
    print(np.array([
        area,
        perimeter,
        width,
        height,
        aspect_ratio,
        mean_rgb[0],
        mean_rgb[1],
        mean_rgb[2]
    ]))
    return np.array([
        area,
        perimeter,
        width,
        height,
        aspect_ratio,
        mean_rgb[0],
        mean_rgb[1],
        mean_rgb[2]
    ])
    
def compare_polygons(poly1, poly2, img1, img2):
    return compare_full_shapes(poly1, poly2, img1, img2)
    
if __name__ == "__main__":
    img1 = "/home/saybrone/2026-01-27-170735_hyprshot.png"
    img2 = "/home/saybrone/2026-01-27-170816_hyprshot.png"

    poly1 = get_artifact_polygon_cm(img1)[0]
    poly2 = get_artifact_polygon_cm(img2)[0]

    print(compare_polygons(poly1, poly2, img1, img2))
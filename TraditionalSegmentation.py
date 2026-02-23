import cv2
import numpy as np


def segment_by_edges(image, min_area=500):
    """Segment using Canny edge detection and contours."""
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(gray.shape, np.uint8)
    
    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            valid_contours.append(cnt)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    return valid_contours, mask


def get_polygon_from_contour(contour):
    """Convert OpenCV contour to polygon list."""
    if contour is None or len(contour) == 0:
        return []
    return contour.squeeze().tolist()


def classify_by_position(contours, img_shape):
    """
    Classify contours as artifact or meter based on position and aspect ratio.
    The meter is typically a long horizontal rectangle.
    """
    if not contours:
        return {}
    
    h, w = img_shape[:2]
    results = {'meter': None, 'artifacts': []}
    
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for i, cnt in enumerate(sorted_contours):
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / ch if ch > 0 else 0
        
        is_horizontal = aspect_ratio > 2
        is_at_edge = x < w * 0.15 or (x + cw) > w * 0.85
        
        if is_horizontal and is_at_edge and results['meter'] is None:
            results['meter'] = cnt
        else:
            results['artifacts'].append(cnt)
    
    if results['meter'] is None and len(sorted_contours) > 0:
        best_meter = None
        best_score = 0
        for cnt in sorted_contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / ch if ch > 0 else 0
            if aspect_ratio > 2:
                score = aspect_ratio
                if score > best_score:
                    best_score = score
                    best_meter = cnt
        
        if best_meter:
            results['meter'] = best_meter
            results['artifacts'] = [c for c in sorted_contours if not np.array_equal(c, best_meter)]
        else:
            results['meter'] = sorted_contours[0]
            results['artifacts'] = sorted_contours[1:]
    
    return results


def separate_objects(image, min_area=500):
    """
    Separate ONE artifact and ONE meter from background using edge detection.
    """
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return {'meter': None, 'artifact': None}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'meter': None, 'artifact': None}
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not valid_contours:
        return {'meter': None, 'artifact': None}
    
    h, w = img.shape[:2]
    
    meter_contour = None
    artifact_contour = None
    
    for cnt in valid_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / ch if ch > 0 else 0
        
        is_horizontal = aspect_ratio > 2
        is_at_edge = x < w * 0.15 or (x + cw) > w * 0.85
        
        if is_horizontal and is_at_edge and meter_contour is None:
            meter_contour = cnt
        elif artifact_contour is None:
            artifact_contour = cnt
    
    if meter_contour is None and len(valid_contours) >= 2:
        for cnt in valid_contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / ch if ch > 0 else 0
            if aspect_ratio > 2 and artifact_contour is not None:
                meter_contour = cnt
                break
    
    return {'meter': meter_contour, 'artifact': artifact_contour}


def calculate_dimensions(contour, px_per_cm):
    """Calculate dimensions of a contour in cm."""
    if contour is None:
        return None
    
    x, y, w, h = cv2.boundingRect(contour)
    
    perimeter_px = cv2.arcLength(contour, True)
    area_px = cv2.contourArea(contour)
    
    w_cm = w / px_per_cm
    h_cm = h / px_per_cm
    perimeter_cm = perimeter_px / px_per_cm
    area_cm2 = area_px / (px_per_cm ** 2)
    
    return {
        'width_cm': round(w_cm, 2),
        'height_cm': round(h_cm, 2),
        'perimeter_cm': round(perimeter_cm, 2),
        'area_cm2': round(area_cm2, 2),
        'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h}
    }


def create_segmentation_visualization(image, classified):
    """Create visualization of segmentation results with dimensions."""
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return None
    
    output = img.copy()
    
    if classified.get('meter') is not None:
        cv2.drawContours(output, [classified['meter']], -1, (0, 255, 255), 3)
        x, y, w, h = cv2.boundingRect(classified['meter'])
        cv2.putText(output, "METER", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if classified.get('artifact') is not None:
        cv2.drawContours(output, [classified['artifact']], -1, (255, 0, 0), 3)
        x, y, w, h = cv2.boundingRect(classified['artifact'])
        cv2.putText(output, "ARTIFACT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return output


def get_all_artifact_polygons(image_path, meter_length_cm=8, min_area=500):
    """
    Main function to extract ONE artifact polygon with dimensions in cm.
    
    Args:
        image_path: Path to the image
        meter_length_cm: Length of the meter in cm (default 8cm)
        min_area: Minimum contour area to consider
    
    Returns:
        Dictionary with meter info and single artifact with dimensions
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    result = separate_objects(image_path, min_area)
    
    meter_contour = result.get('meter')
    if meter_contour is None:
        return {'error': 'No meter detected. Ensure image contains a rectangular meter.', 'meter': None, 'artifact': None}
    
    mx, my, mw, mh = cv2.boundingRect(meter_contour)
    
    if mw > mh:
        meter_pixel_length = mw
    else:
        meter_pixel_length = mh
    
    if meter_pixel_length == 0:
        meter_pixel_length = 1
    
    px_per_cm = meter_pixel_length / meter_length_cm
    
    artifact_contour = result.get('artifact')
    artifact_info = None
    
    if artifact_contour is not None:
        poly = get_polygon_from_contour(artifact_contour)
        if poly and len(poly) >= 3:
            poly_cm = [[pt[0] / px_per_cm, pt[1] / px_per_cm] for pt in poly]
            dims = calculate_dimensions(artifact_contour, px_per_cm)
            artifact_info = {
                'polygon': poly_cm,
                'polygon_px': poly,
                'dimensions': dims,
                'contour': artifact_contour
            }
    
    meter_dims = calculate_dimensions(meter_contour, px_per_cm)
    
    return {
        'meter': {
            'contour': meter_contour,
            'dimensions': meter_dims,
            'polygon_px': get_polygon_from_contour(meter_contour)
        },
        'artifact': artifact_info,
        'scale': {
            'pixels_per_cm': px_per_cm,
            'meter_length_cm': meter_length_cm
        }
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python TraditionalSegmentation.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Processing {image_path}...")
    print(f"Looking for meter: 8cm rectangular object")
    print("-" * 50)
    
    result = get_all_artifact_polygons(image_path, meter_length_cm=8, min_area=500)
    
    if result is None:
        print("Error: Could not read image")
        sys.exit(1)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print(f"Scale: {result['scale']['pixels_per_cm']:.2f} pixels per cm")
    print()
    
    if result['meter']:
        m = result['meter']
        print(f"METER DETECTED:")
        print(f"  Width:  {m['dimensions']['width_cm']} cm")
        print(f"  Height: {m['dimensions']['height_cm']} cm")
        print(f"  Perimeter: {m['dimensions']['perimeter_cm']} cm")
        print(f"  Area: {m['dimensions']['area_cm2']} cm²")
        print()
    
    if result['artifact']:
        art = result['artifact']
        d = art['dimensions']
        print(f"ARTIFACT DETECTED:")
        print(f"  Width:      {d['width_cm']} cm")
        print(f"  Height:     {d['height_cm']} cm")
        print(f"  Perimeter: {d['perimeter_cm']} cm")
        print(f"  Area:       {d['area_cm2']} cm²")
    else:
        print("No artifact detected")
    
    vis = create_segmentation_visualization(image_path, {'meter': result.get('meter', {}).get('contour'), 'artifact': result.get('artifact', {}).get('contour')})
    
    output_path = image_path.rsplit('.', 1)[0] + '_segmented.jpg'
    cv2.imwrite(output_path, vis)
    print(f"Saved visualization to {output_path}")


def extract_color_features(img, contour):
    """Extract color features from the region inside a contour."""
    if img is None or contour is None:
        return None
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    
    mean_color = cv2.mean(img, mask=mask)[:3]
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
    
    hist_bins = 32
    hist = cv2.calcHist([img], [0, 1, 2], mask, [hist_bins, hist_bins, hist_bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return {
        'mean_bgr': mean_color,
        'mean_hsv': mean_hsv,
        'histogram': hist
    }


def color_similarity(color1, color2):
    """Calculate color similarity0-100 between two artifacts ()."""
    if color1 is None or color2 is None:
        return 0
    
    bgr_dist = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1['mean_bgr'], color2['mean_bgr'])))
    bgr_similarity = max(0, 100 - bgr_dist / 4.41 * 100)
    
    hsv_dist = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1['mean_hsv'], color2['mean_hsv'])))
    hsv_similarity = max(0, 100 - hsv_dist / 180 * 100)
    
    hist_similarity = 0
    if color1['histogram'] is not None and color2['histogram'] is not None:
        hist_similarity = (1 - cv2.compareHist(color1['histogram'], color2['histogram'], cv2.HISTCMP_CORREL)) * 100
    
    return (bgr_similarity * 0.3 + hsv_similarity * 0.3 + hist_similarity * 0.4)


def extract_edge_profile(contour, num_points=100):
    """Extract edge profile from contour for puzzle fit analysis."""
    if contour is None or len(contour) < 3:
        return None
    
    contour = contour.squeeze() if len(contour.shape) == 3 else contour
    
    if len(contour) < 3:
        return None
    
    points = contour.astype(np.float32)
    
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
    distances = distances / distances[-1] if distances[-1] > 0 else distances
    
    x_interp = np.interp(np.linspace(0, 1, num_points), distances, points[:, 0])
    y_interp = np.interp(np.linspace(0, 1, num_points), distances, points[:, 1])
    
    angles = np.arctan2(np.diff(y_interp, append=y_interp[0]), np.diff(x_interp, append=x_interp[0]))
    angles = np.mod(angles + np.pi, 2 * np.pi)
    
    curvatures = []
    for i in range(len(angles)):
        prev_angle = angles[i - 1]
        curr_angle = angles[i]
        curvature = np.abs(np.mod(curr_angle - prev_angle + np.pi, 2 * np.pi) - np.pi)
        curvatures.append(curvature)
    
    return {
        'points': np.column_stack([x_interp, y_interp]),
        'angles': angles,
        'curvatures': np.array(curvatures),
        'centroid': np.mean(points, axis=0)
    }


def analyze_edge_features(poly):
    """Analyze edge to find convex and concave segments."""
    if poly is None or len(poly) < 3:
        return None
    
    poly = np.array(poly)
    
    edge_features = {
        'left_edge': [],
        'right_edge': [],
        'top_edge': [],
        'bottom_edge': []
    }
    
    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    w, h = x_max - x_min, y_max - y_min
    
    for i in range(len(poly)):
        x, y = poly[i]
        
        rel_x = (x - x_min) / w if w > 0 else 0.5
        rel_y = (y - y_min) / h if h > 0 else 0.5
        
        next_i = (i + 1) % len(poly)
        prev_i = (i - 1) % len(poly)
        
        v1 = poly[prev_i] - poly[i]
        v2 = poly[next_i] - poly[i]
        
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        curvature = cross / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        
        if rel_x < 0.3:
            edge_features['left_edge'].append({'y': rel_y, 'curvature': curvature, 'point': (x, y)})
        elif rel_x > 0.7:
            edge_features['right_edge'].append({'y': rel_y, 'curvature': curvature, 'point': (x, y)})
        
        if rel_y < 0.3:
            edge_features['top_edge'].append({'x': rel_x, 'curvature': curvature, 'point': (x, y)})
        elif rel_y > 0.7:
            edge_features['bottom_edge'].append({'x': rel_x, 'curvature': curvature, 'point': (x, y)})
    
    return {
        'poly': poly,
        'bounds': {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max, 'w': w, 'h': h},
        'features': edge_features
    }


def find_connection_points(edge1, edge2):
    """Find best matching points between two edges."""
    if not edge1 or not edge2:
        return None
    
    matches = []
    
    for p1 in edge1:
        for p2 in edge2:
            y_diff = abs(p1['y'] - p2['y']) if 'y' in p1 else abs(p1['x'] - p2['x'])
            curv_compat = p1['curvature'] * p2['curvature'] < 0
            
            if y_diff < 0.2:
                score = (1 - y_diff) * 100
                if curv_compat:
                    score += 30
                matches.append({
                    'point1': p1['point'],
                    'point2': p2['point'],
                    'score': score,
                    'curvature1': p1['curvature'],
                    'curvature2': p2['curvature']
                })
    
    if matches:
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[0]
    
    return None


def analyze_connections(artifacts_data):
    """Analyze how multiple artifacts connect."""
    import json
    
    analyzed = []
    
    for i, art in enumerate(artifacts_data):
        poly = json.loads(art['polygon'])
        analyzed.append(analyze_edge_features(poly))
    
    connections = []
    
    for i in range(len(analyzed)):
        art1 = analyzed[i]
        art2 = analyzed[(i + 1) % len(analyzed)]
        
        if art1 is None or art2 is None:
            connections.append(None)
            continue
        
        left_edge = art1['features']['left_edge']
        right_edge = art1['features']['right_edge']
        
        right_match = find_connection_points(right_edge, art2['features']['left_edge']) if right_edge and art2['features']['left_edge'] else None
        left_match = find_connection_points(left_edge, art2['features']['right_edge']) if left_edge and art2['features']['right_edge'] else None
        
        match = right_match if right_match and (not left_match or right_match['score'] > left_match['score']) else left_match
        
        connections.append({
            'from_idx': i,
            'to_idx': (i + 1) % len(analyzed),
            'match': match,
            'art1_bounds': art1['bounds'],
            'art2_bounds': art2['bounds']
        })
    
    return connections, analyzed


def transform_piece_to_connect(poly1, poly2, connection_info):
    """Transform poly2 to connect with poly1 based on matching edges."""
    if poly1 is None or poly2 is None or connection_info is None:
        return poly2
    
    match = connection_info.get('match')
    if match is None:
        return poly2
    
    poly1 = np.array(poly1)
    poly2 = np.array(poly2)
    
    p1 = match.get('point1')
    p2 = match.get('point2')
    
    if p1 is None or p2 is None:
        return poly2
    
    x1_min, y1_min = poly1.min(axis=0)
    x1_max, y1_max = poly1.max(axis=0)
    x2_min, y2_min = poly2.min(axis=0)
    x2_max, y2_max = poly2.max(axis=0)
    
    target_x = p1[0]
    source_x = p2[0]
    dx = target_x - source_x
    
    target_y = p1[1]
    source_y = p2[1]
    dy = target_y - source_y
    
    translated = poly2.copy()
    translated[:, 0] = translated[:, 0] + dx
    translated[:, 1] = translated[:, 1] + dy
    
    return translated


def edge_complementarity(poly1, poly2):
    """
    Calculate puzzle fit score based on:
    1. Bounding box compatibility - can they fit together?
    2. Edge profile matching - do edges complement each other?
    3. Size ratio - are sizes compatible?
    """
    if poly1 is None or poly2 is None:
        return 0
    
    try:
        poly1 = np.array(poly1)
        poly2 = np.array(poly2)
        
        if len(poly1) < 3 or len(poly2) < 3:
            return 0
        
        x1_min, y1_min = poly1.min(axis=0)
        x1_max, y1_max = poly1.max(axis=0)
        x2_min, y2_min = poly2.min(axis=0)
        x2_max, y2_max = poly2.max(axis=0)
        
        w1, h1 = x1_max - x1_min, y1_max - y1_min
        w2, h2 = x2_max - x2_min, y2_max - y2_min
        
        area1 = w1 * h1
        area2 = w2 * h2
        
        if area1 <= 0 or area2 <= 0:
            return 0
        
        size_ratio = min(area1, area2) / max(area1, area2)
        size_score = size_ratio * 100
        
        def get_edge_profile(poly):
            points = poly.astype(np.float32)
            cx, cy = points[:, 0].mean(), points[:, 1].mean()
            
            centered = points - np.array([cx, cy])
            angles = np.arctan2(centered[:, 1], centered[:, 0])
            
            sorted_indices = np.argsort(angles)
            sorted_points = centered[sorted_indices]
            
            distances = np.sqrt(sorted_points[:, 0]**2 + sorted_points[:, 1]**2)
            
            resampled = np.interp(
                np.linspace(0, 1, 50),
                np.linspace(0, 1, len(distances)),
                distances
            )
            
            return resampled
        
        profile1 = get_edge_profile(poly1)
        profile2 = get_edge_profile(poly2)
        
        max_corr = 0
        for shift in range(25):
            shifted = np.roll(profile2, shift)
            corr = np.corrcoef(profile1, shifted)[0, 1]
            if not np.isnan(corr):
                max_corr = max(max_corr, abs(corr))
        
        edge_score = max_corr * 100
        
        combined_width = max(w1, w2) + min(w1, w2) * 0.3
        combined_height = max(h1, h2) + min(h1, h2) * 0.3
        
        perimeter1 = np.sum(np.linalg.norm(np.diff(poly1, axis=0), axis=1))
        perimeter2 = np.sum(np.linalg.norm(np.diff(poly2, axis=0), axis=1))
        
        combined_perimeter = perimeter1 + perimeter2
        joined_perimeter = 2 * (min(w1, w2) + min(h1, h2))
        
        if combined_perimeter > 0:
            join_efficiency = min(joined_perimeter / combined_perimeter * 100, 100)
        else:
            join_efficiency = 0
        
        total_score = (size_score * 0.3 + edge_score * 0.4 + join_efficiency * 0.3)
        
        return max(0, min(100, total_score))
        
    except Exception as e:
        print(f"Puzzle fit error: {e}")
        return 0


def pattern_similarity(poly1, poly2):
    """Calculate shape/pattern similarity (0-100)."""
    if not poly1 or not poly2:
        return 0
    
    poly1 = np.array(poly1)
    poly2 = np.array(poly2)
    
    if len(poly1) < 3 or len(poly2) < 3:
        return 0
    
    c1 = np.mean(poly1, axis=0)
    c2 = np.mean(poly2, axis=0)
    
    poly1_centered = poly1 - c1
    poly2_centered = poly2 - c2
    
    scale1 = np.max(np.linalg.norm(poly1_centered, axis=1))
    scale2 = np.max(np.linalg.norm(poly2_centered, axis=1))
    
    if scale1 > 0:
        poly1_normalized = poly1_centered / scale1
    else:
        poly1_normalized = poly1_centered
    
    if scale2 > 0:
        poly2_normalized = poly2_centered / scale2
    else:
        poly2_normalized = poly2_centered
    
    num_points = 64
    distances1 = np.zeros(len(poly1_normalized))
    for i in range(1, len(poly1_normalized)):
        distances1[i] = distances1[i-1] + np.linalg.norm(poly1_normalized[i] - poly1_normalized[i-1])
    distances1 = distances1 / distances1[-1] if distances1[-1] > 0 else distances1
    
    distances2 = np.zeros(len(poly2_normalized))
    for i in range(1, len(poly2_normalized)):
        distances2[i] = distances2[i-1] + np.linalg.norm(poly2_normalized[i] - poly2_normalized[i-1])
    distances2 = distances2 / distances2[-1] if distances2[-1] > 0 else distances2
    
    x1 = np.interp(np.linspace(0, 1, num_points), distances1, poly1_normalized[:, 0])
    y1 = np.interp(np.linspace(0, 1, num_points), distances1, poly1_normalized[:, 1])
    x2 = np.interp(np.linspace(0, 1, num_points), distances2, poly2_normalized[:, 0])
    y2 = np.interp(np.linspace(0, 1, num_points), distances2, poly2_normalized[:, 1])
    
    euclidean_dist = np.sqrt(np.sum((x1 - x2) ** 2 + (y1 - y2) ** 2)) / num_points
    
    perimeter1 = np.sum(np.linalg.norm(np.diff(poly1_normalized, axis=0), axis=1))
    perimeter2 = np.sum(np.linalg.norm(np.diff(poly2_normalized, axis=0), axis=1))
    perimeter_ratio = min(perimeter1, perimeter2) / max(perimeter1, perimeter2) if max(perimeter1, perimeter2) > 0 else 0
    
    area1 = cv2.contourArea(poly1_normalized.astype(np.float32))
    area2 = cv2.contourArea(poly2_normalized.astype(np.float32))
    area_ratio = min(abs(area1), abs(area2)) / max(abs(area1), abs(area2)) if max(abs(area1), abs(area2)) > 0 else 0
    
    shape_score = (1 - min(euclidean_dist * 5, 1)) * 50 + perimeter_ratio * 25 + area_ratio * 25
    
    return max(0, min(100, shape_score))


def compare_artifacts(img1_path, img2_path, poly1, poly2):
    """
    Comprehensive artifact similarity comparison considering:
    - Color (30%)
    - Pattern/shape (30%)
    - Puzzle fit / edge complementarity (40%)
    
    Returns similarity score 0-100
    """
    contour1 = np.array(poly1, dtype=np.int32) if poly1 else None
    contour2 = np.array(poly2, dtype=np.int32) if poly2 else None
    
    img1 = cv2.imread(img1_path) if isinstance(img1_path, str) else img1_path
    img2 = cv2.imread(img2_path) if isinstance(img2_path, str) else img2_path
    
    color1 = extract_color_features(img1, contour1)
    color2 = extract_color_features(img2, contour2)
    color_score = color_similarity(color1, color2)
    
    pattern_score = pattern_similarity(poly1, poly2)
    
    puzzle_score = edge_complementarity(poly1, poly2)
    
    weights = {
        'color': 0.30,
        'pattern': 0.30,
        'puzzle': 0.40
    }
    
    final_score = (
        color_score * weights['color'] +
        pattern_score * weights['pattern'] +
        puzzle_score * weights['puzzle']
    )
    
    return round(final_score, 2)

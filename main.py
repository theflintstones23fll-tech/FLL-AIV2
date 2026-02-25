from flask import Flask, request, jsonify, render_template, session, redirect, url_for, g
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tempfile
from PIL import Image
import io
import base64
import sqlite3
import hashlib
import secrets
from functools import wraps
from datetime import datetime

from TraditionalSegmentation import separate_objects, classify_by_position, create_segmentation_visualization, get_polygon_from_contour, get_all_artifact_polygons, calculate_dimensions, compare_artifacts
from fracture_matching import find_connection, visualise_assembly, get_artifact_contour

METER_LENGTH_CM = 8

def mark_connection_on_artifacts(img1_path, img2_path, result, pair_num):
    """
    Show two artifacts assembled together aligned to their common vessel circle.
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return None
    
    intern = result.get('internal')
    if not intern:
        return None
    
    tf = intern.get('transform')
    if tf is None:
        return None
    
    si, sj = intern.get('best_pair', (0, 0))
    segs1 = intern.get('segs1', [])
    segs2 = intern.get('segs2', [])
    pts1 = intern.get('pts1')
    pts2 = intern.get('pts2')
    
    if pts1 is None or pts2 is None:
        return None
    
    dx = tf.get('dx', 0)
    dy = tf.get('dy', 0)
    pts2_t = pts2 + np.array([dx, dy])
    
    all_pts = np.vstack([pts1, pts2_t])
    margin = 40
    x_min, y_min = all_pts.min(axis=0) - margin
    x_max, y_max = all_pts.max(axis=0) + margin
    
    scale = min(800 / (x_max - x_min), 500 / (y_max - y_min), 1.5)
    CW = int((x_max - x_min) * scale) + 60
    CH = int((y_max - y_min) * scale) + 80
    canvas = np.full((CH, CW, 3), 25, dtype=np.uint8)
    ox = int(-x_min * scale + 30)
    oy = int(-y_min * scale + 50)
    
    def to_canvas(pts):
        return (pts * scale + np.array([ox, oy])).astype(np.int32)
    
    def blit(img_src, pts_orig, pts_c, outline_col):
        sx, sy, sw, sh = cv2.boundingRect(pts_orig.astype(np.int32))
        sx1, sy1 = max(0, sx - 2), max(0, sy - 2)
        sx2, sy2 = min(img_src.shape[1], sx + sw + 2), min(img_src.shape[0], sy + sh + 2)
        src = img_src[sy1:sy2, sx1:sx2]
        if src.size == 0:
            return
        sw_s, sh_s = int((sx2 - sx1) * scale), int((sy2 - sy1) * scale)
        if sw_s < 1 or sh_s < 1:
            return
        src_r = cv2.resize(src, (sw_s, sh_s))
        mask = np.zeros((sh_s, sw_s), dtype=np.uint8)
        lpts = ((pts_orig - np.array([sx1, sy1])) * scale).astype(np.int32)
        cv2.fillPoly(mask, [lpts], 255)
        
        bx, by, _, _ = cv2.boundingRect(pts_c)
        dst_x, dst_y = bx, by
        d_x2 = min(canvas.shape[1], dst_x + sw_s)
        d_y2 = min(canvas.shape[0], dst_y + sh_s)
        s_x2, s_y2 = min(sw_s, d_x2 - dst_x), min(sh_s, d_y2 - dst_y)
        if s_x2 <= 0 or s_y2 <= 0:
            return
        patch = src_r[:s_y2, :s_x2]
        msk = mask[:s_y2, :s_x2]
        reg = canvas[dst_y:dst_y + s_y2, dst_x:dst_x + s_x2]
        if patch.shape != reg.shape:
            return
        canvas[dst_y:dst_y + s_y2, dst_x:dst_x + s_x2] = np.where(
            msk[:, :, None] > 0, patch, reg)
        cv2.drawContours(canvas, [pts_c.reshape(-1, 1, 2)], -1, outline_col, 2)
    
    blit(img1, pts1, to_canvas(pts1), (80, 200, 255))
    blit(img2, pts2, to_canvas(pts2_t), (80, 255, 150))
    
    if si < len(segs1):
        arc1 = to_canvas(segs1[si]['points'])
        cv2.polylines(canvas, [arc1], False, (0, 255, 255), 3)
        for pt in arc1:
            cv2.circle(canvas, tuple(pt), 4, (0, 255, 255), -1)
    
    if sj < len(segs2):
        arc2 = to_canvas(segs2[sj]['points'] + np.array([dx, dy]))
        cv2.polylines(canvas, [arc2], False, (0, 255, 0), 3)
        for pt in arc2:
            cv2.circle(canvas, tuple(pt), 4, (0, 255, 0), -1)
    
    if si < len(segs1) and sj < len(segs2):
        arc1 = segs1[si]['points']
        arc2 = segs2[sj]['points'] + np.array([dx, dy])
        
        closest_p1, closest_p2 = None, None
        min_dist = float('inf')
        
        for p1 in arc1:
            for p2 in arc2:
                d = np.linalg.norm(p1 - p2)
                if d < min_dist:
                    min_dist = d
                    closest_p1 = p1
                    closest_p2 = p2
        
        if closest_p1 is not None and closest_p2 is not None:
            cp1 = to_canvas(closest_p1.reshape(1, -1))[0]
            cp2 = to_canvas(closest_p2.reshape(1, -1))[0]
            cv2.circle(canvas, tuple(cp1), 12, (0, 165, 255), 3)
            cv2.circle(canvas, tuple(cp2), 12, (0, 165, 255), 3)
    
    return canvas

def get_artifact_polygon_cm(img_path, meter_length_cm=METER_LENGTH_CM):
    result = get_all_artifact_polygons(img_path, meter_length_cm=meter_length_cm, min_area=500)
    
    if result is None or 'error' in result:
        return None
    
    artifact = result.get('artifact')
    if artifact is None:
        return None
    
    return [artifact['polygon']]


def save_segmentation_image(img_path, output_path=None):
    result = separate_objects(img_path, min_area=500)
    
    vis = create_segmentation_visualization(img_path, result)
    
    if vis is None:
        return img_path
    
    if output_path is None:
        output_path = img_path.rsplit('.', 1)[0] + '_result.jpg'
    
    cv2.imwrite(output_path, vis)
    return output_path


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = secrets.token_hex(32)

DATABASE = 'artifacts.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_base64 TEXT,
                polygon TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                origin TEXT,
                era TEXT,
                result_image TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        try:
            cursor.execute('ALTER TABLE artifacts ADD COLUMN result_image TEXT')
        except sqlite3.OperationalError:
            pass
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS artifact_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id INTEGER NOT NULL,
                area REAL,
                perimeter REAL,
                width REAL,
                height REAL,
                aspect_ratio REAL,
                mean_r REAL,
                mean_g REAL,
                mean_b REAL,
                FOREIGN KEY (artifact_id) REFERENCES artifacts (id)
            )
        ''')
        
        db.commit()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password, password_hash):
    return hashlib.sha256(password.encode()).hexdigest() == password_hash

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    filename = secure_filename(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
        file.save(tmp_file.name)
        return tmp_file.name

def image_to_base64(img_path):
    with Image.open(img_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

def get_user_artifacts(user_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT id, name, image_path, image_base64, polygon, created_at, 
               description, origin, era, result_image
        FROM artifacts WHERE user_id = ? ORDER BY created_at DESC
    ''', (user_id,))
    artifacts = []
    for row in cursor.fetchall():
        artifacts.append({
            'id': row['id'],
            'name': row['name'],
            'image_path': row['image_path'],
            'image_base64': row['image_base64'],
            'polygon': row['polygon'],
            'timestamp': row['created_at'],
            'description': row['description'],
            'origin': row['origin'],
            'era': row['era'],
            'result_image': row['result_image']
        })
    return artifacts

def get_artifact_by_id(artifact_id, user_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        SELECT id, name, image_path, image_base64, polygon, created_at,
               description, origin, era, result_image
        FROM artifacts WHERE id = ? AND user_id = ?
    ''', (artifact_id, user_id))
    row = cursor.fetchone()
    if row:
        cursor.execute('''
            SELECT area, perimeter, width, height, aspect_ratio, 
                   mean_r, mean_g, mean_b
            FROM artifact_features WHERE artifact_id = ?
        ''', (artifact_id,))
        features = cursor.fetchone()
        return {
            'id': row['id'],
            'name': row['name'],
            'image_path': row['image_path'],
            'image_base64': row['image_base64'],
            'polygon': row['polygon'],
            'timestamp': row['created_at'],
            'description': row['description'],
            'origin': row['origin'],
            'era': row['era'],
            'result_image': row['result_image'],
            'features': dict(features) if features else None
        }
    return None

@app.route('/')
def index():
    user_id = session.get('user_id')
    artifacts = get_user_artifacts(user_id) if user_id else []
    return render_template('index.html', artifacts=artifacts, user_logged_in=bool(user_id), username=session.get('username'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
        else:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
        
        if not username or not email or not password:
            return jsonify({'error': 'All fields are required'}), 400
        
        db = get_db()
        cursor = db.cursor()
        
        try:
            password_hash = hash_password(password)
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            db.commit()
            return jsonify({'message': 'Registration successful! Please log in.'}), 201
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Username or email already exists'}), 400
    
    return render_template('index.html', show_register=True)

@app.route('/login', methods=['POST'])
def login():
    if request.is_json:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
    else:
        username = request.form.get('username')
        password = request.form.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT id, username, password_hash FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    
    if user and check_password(password, user['password_hash']):
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({'message': 'Login successful', 'username': user['username']}), 200
    
    return jsonify({'error': 'Username or password is incorrect'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/check_auth', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        return jsonify({'authenticated': True, 'username': session.get('username')}), 200
    return jsonify({'authenticated': False}), 200


@app.route('/add_to_database', methods=['POST'])
@login_required
def add_to_database():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        artifact_name = request.form.get('name', f'Artifact_{datetime.now().strftime("%Y%m%d%H%M%S")}')
        description = request.form.get('description', '')
        origin = request.form.get('origin', '')
        era = request.form.get('era', '')
        
        filename = secure_filename(file.filename)
        artifact_path = save_uploaded_file(file)
        
        artifact_polygons = get_artifact_polygon_cm(artifact_path)
        if not artifact_polygons or len(artifact_polygons) == 0:
            os.unlink(artifact_path)
            return jsonify({'error': 'No artifacts detected in the image'}), 400
        
        artifact_polygon_cm = artifact_polygons[0]
        image_base64 = image_to_base64(artifact_path)
        
        result_image_path = save_segmentation_image(artifact_path)
        result_image_base64 = image_to_base64(result_image_path)
        
        try:
            os.unlink(result_image_path)
        except:
            pass
        
        import json
        polygon_json = json.dumps(artifact_polygons[0])
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO artifacts (user_id, name, image_path, image_base64, polygon, description, origin, era, result_image)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], artifact_name, artifact_path, image_base64, polygon_json, description, origin, era, result_image_base64))
        
        artifact_id = cursor.lastrowid
        
        try:
            from Segmentation import polygon_features
            features = polygon_features(artifact_polygon_cm, artifact_path)
            cursor.execute('''
                INSERT INTO artifact_features (artifact_id, area, perimeter, width, height, aspect_ratio, mean_r, mean_g, mean_b)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (artifact_id, features[0], features[1], features[2], features[3], features[4], features[5], features[6], features[7]))
        except Exception:
            pass
        
        db.commit()
        
        return jsonify({'message': f'Artifact "{artifact_name}" added successfully', 'artifact_id': artifact_id}), 200
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/artifact/<int:artifact_id>', methods=['GET'])
@login_required
def view_artifact(artifact_id):
    artifact = get_artifact_by_id(artifact_id, session['user_id'])
    if not artifact:
        return jsonify({'error': 'Artifact not found'}), 404
    
    import json
    artifact_polygon = json.loads(artifact['polygon'])
    
    user_artifacts = get_user_artifacts(session['user_id'])
    similarities = []
    
    for hist_artifact in user_artifacts:
        if hist_artifact['id'] == artifact_id:
            continue
        
        try:
            hist_polygon = json.loads(hist_artifact['polygon'])
            similarity = compare_artifacts(
                artifact['image_path'],
                hist_artifact['image_path'],
                artifact_polygon,
                hist_polygon
            )
            similarities.append({
                'id': hist_artifact['id'],
                'name': hist_artifact['name'],
                'similarity_score': float(similarity),
                'image_base64': hist_artifact['image_base64'],
                'result_image': hist_artifact.get('result_image'),
                'timestamp': hist_artifact['timestamp']
            })
        except:
            continue
    
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    artifact['similar_artifacts'] = similarities[:3]
    artifact['all_similar_count'] = len(similarities)
    
    return jsonify(artifact), 200


@app.route('/artifact/<int:artifact_id>/similar', methods=['GET'])
@login_required
def get_similar_artifacts(artifact_id):
    artifact = get_artifact_by_id(artifact_id, session['user_id'])
    if not artifact:
        return jsonify({'error': 'Artifact not found'}), 404
    
    offset = request.args.get('offset', 0, type=int)
    limit = request.args.get('limit', 5, type=int)
    
    import json
    artifact_polygon = json.loads(artifact['polygon'])
    
    user_artifacts = get_user_artifacts(session['user_id'])
    similarities = []
    
    for hist_artifact in user_artifacts:
        if hist_artifact['id'] == artifact_id:
            continue
        
        try:
            hist_polygon = json.loads(hist_artifact['polygon'])
            similarity = compare_artifacts(
                artifact['image_path'],
                hist_artifact['image_path'],
                artifact_polygon,
                hist_polygon
            )
            similarities.append({
                'id': hist_artifact['id'],
                'name': hist_artifact['name'],
                'similarity_score': float(similarity),
                'image_base64': hist_artifact['image_base64'],
                'result_image': hist_artifact.get('result_image'),
                'timestamp': hist_artifact['timestamp']
            })
        except:
            continue
    
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return jsonify({
        'artifacts': similarities[offset:offset + limit],
        'total': len(similarities),
        'offset': offset,
        'has_more': len(similarities) > offset + limit
    }), 200

@app.route('/artifact/<int:artifact_id>', methods=['DELETE'])
@login_required
def delete_artifact(artifact_id):
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('SELECT image_path FROM artifacts WHERE id = ? AND user_id = ?', (artifact_id, session['user_id']))
    row = cursor.fetchone()
    
    if not row:
        return jsonify({'error': 'Artifact not found'}), 404
    
    try:
        os.unlink(row['image_path'])
    except:
        pass
    
    cursor.execute('DELETE FROM artifact_features WHERE artifact_id = ?', (artifact_id,))
    cursor.execute('DELETE FROM artifacts WHERE id = ? AND user_id = ?', (artifact_id, session['user_id']))
    db.commit()
    
    return jsonify({'message': 'Artifact deleted successfully'}), 200

@app.route('/database', methods=['GET'])
@login_required
def view_database():
    artifacts = get_user_artifacts(session['user_id'])
    return jsonify({
        'count': len(artifacts),
        'artifacts': [
            {
                'id': a['id'],
                'name': a['name'],
                'timestamp': a['timestamp'],
                'description': a['description'],
                'origin': a['origin'],
                'era': a['era']
            }
            for a in artifacts
        ]
    })
    

@app.route('/compare_artifacts', methods=['POST'])
@login_required
def compare_two_artifacts():
    data = request.get_json()
    artifact1_id = data.get('artifact1_id')
    artifact2_id = data.get('artifact2_id')
    
    if not artifact1_id or not artifact2_id:
        return jsonify({'error': 'Two artifact IDs required'}), 400
    
    if artifact1_id == artifact2_id:
        return jsonify({'error': 'Cannot compare same artifact'}), 400
    
    artifact1 = get_artifact_by_id(artifact1_id, session['user_id'])
    artifact2 = get_artifact_by_id(artifact2_id, session['user_id'])
    
    if not artifact1 or not artifact2:
        return jsonify({'error': 'One or both artifacts not found'}), 404
    
    try:
        result = find_connection(artifact1['image_path'], artifact2['image_path'])
        puzzle_score = result.get('score', 0)
        connects = result.get('connects', False)
        
        assembly_image = None
        if result.get('internal'):
            annotated = mark_connection_on_artifacts(
                artifact1['image_path'], artifact2['image_path'], 
                result, 1
            )
            if annotated is not None:
                _, buffer = cv2.imencode('.png', annotated)
                assembly_image = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        final_score = puzzle_score
        
        verdict = "Perfect Match!"
        if final_score < 20:
            verdict = "May Not Fit Together"
        elif final_score < 40:
            verdict = "Low Compatibility"
        elif final_score < 60:
            verdict = "Moderate Match"
        elif final_score < 80:
            verdict = "Good Fit"
        
        return jsonify({
            'artifact1': {
                'id': artifact1['id'],
                'name': artifact1['name'],
                'image_base64': artifact1['image_base64']
            },
            'artifact2': {
                'id': artifact2['id'],
                'name': artifact2['name'],
                'image_base64': artifact2['image_base64']
            },
            'scores': {
                'puzzle_fit': float(round(puzzle_score, 2)),
                'total': float(round(final_score, 2))
            },
            'verdict': verdict,
            'connects': connects,
            'assembly_image': assembly_image,
            'fracture_message': result.get('message', '')
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500


def create_connection_visualization(artifacts, pair_scores, connection_points):
    """Create an image showing artifacts as a single connected shape using edge transformation."""
    import json
    from TraditionalSegmentation import analyze_edge_features, find_connection_points, transform_piece_to_connect
    
    if len(artifacts) == 0:
        return None
    
    canvas_height = 700
    canvas_width = 1000
    
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    colors = [
        (200, 120, 120),
        (120, 200, 120),
        (120, 120, 200),
        (200, 200, 120),
        (200, 120, 200),
        (120, 200, 200)
    ]
    
    transformed_polys = []
    
    for i, art in enumerate(artifacts):
        poly = json.loads(art['polygon'])
        poly_arr = np.array(poly, dtype=np.float64)
        transformed_polys.append(poly_arr)
    
    connections = []
    for i in range(len(artifacts)):
        can_connect = pair_scores[i]['can_connect'] if i < len(pair_scores) else False
        
        if can_connect and i < len(artifacts):
            poly1 = json.loads(artifacts[i]['polygon'])
            poly2 = json.loads(artifacts[(i + 1) % len(artifacts)]['polygon'])
            
            edge1 = analyze_edge_features(poly1)
            edge2 = analyze_edge_features(poly2)
            
            match = None
            if edge1 and edge2:
                right_edge = edge1['features'].get('right_edge', [])
                left_edge = edge2['features'].get('left_edge', [])
                if right_edge and left_edge:
                    match = find_connection_points(right_edge, left_edge)
            
            if not match and edge1 and edge2:
                left_edge = edge1['features'].get('left_edge', [])
                right_edge = edge2['features'].get('right_edge', [])
                if left_edge and right_edge:
                    match = find_connection_points(left_edge, right_edge)
            
            if match:
                connections.append({
                    'from_idx': i,
                    'to_idx': (i + 1) % len(artifacts),
                    'match': match,
                    'can_connect': can_connect
                })
                
                transformed = transform_piece_to_connect(
                    np.array(poly1),
                    np.array(poly2),
                    {'match': match}
                )
                transformed_polys[(i + 1) % len(artifacts)] = transformed
            else:
                connections.append({'can_connect': False})
        else:
            connections.append({'can_connect': can_connect})
    
    all_contours = []
    
    max_width = 0
    max_height = 0
    for poly_arr in transformed_polys:
        poly_arr = np.array(poly_arr)
        w = poly_arr[:, 0].max() - poly_arr[:, 0].min()
        h = poly_arr[:, 1].max() - poly_arr[:, 1].min()
        max_width = max(max_width, w)
        max_height = max(max_height, h)
    
    scale = min((canvas_width - 100) / (max_width * len(artifacts)), (canvas_height - 200) / max_height) * 0.8
    scale = max(scale, 2.0)
    
    for i, poly_arr in enumerate(transformed_polys):
        poly_arr = np.array(poly_arr, dtype=np.float64)
        
        cx = poly_arr[:, 0].mean()
        cy = poly_arr[:, 1].mean()
        
        scaled = poly_arr.copy()
        scaled[:, 0] = (scaled[:, 0] - cx) * scale + cx
        scaled[:, 1] = (scaled[:, 1] - cy) * scale + cy
        
        poly_int = np.array(scaled, dtype=np.int32)
        
        x_min = poly_int[:, 0].min()
        y_min = poly_int[:, 1].min()
        
        if i == 0:
            offset_x = 80
            offset_y = canvas_height // 2
        else:
            prev_poly = all_contours[-1]
            if len(prev_poly) > 0:
                prev_bounds = prev_poly.reshape(-1, 2)
                prev_right = prev_bounds[:, 0].max()
                
                offset_x = int(prev_right) - int(x_min) + 5
                offset_y = canvas_height // 2
            else:
                offset_x = 80 + i * 150
                offset_y = canvas_height // 2
        
        poly_int[:, 0] = np.clip(poly_int[:, 0] - x_min + offset_x, 0, canvas_width - 1)
        poly_int[:, 1] = np.clip(poly_int[:, 1] - y_min + offset_y, 0, canvas_height - 1)
        
        all_contours.append(poly_int)
    
    if len(all_contours) >= 1:
        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        
        for i, cnt in enumerate(all_contours):
            cv2.fillPoly(mask, [cnt.astype(np.int32)], 255)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        merged_color = (160, 160, 160)
        for cnt in contours:
            cv2.fillPoly(canvas, [cnt], merged_color)
            cv2.polylines(canvas, [cnt], True, (60, 60, 60), 3)
        
        for i, cnt in enumerate(all_contours):
            bounds = cnt.reshape(-1, 2)
            cx = int(bounds[:, 0].mean())
            cy = int(bounds[:, 1].mean())
            cv2.putText(canvas, f"#{i+1}", (cx - 15, cy + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if i > 0 and i - 1 < len(connections):
                if not connections[i - 1].get('can_connect', False):
                    edge_x = int(bounds[:, 0].min()) + 5
                    edge_y = int(bounds[:, 1].mean())
                    cv2.putText(canvas, "X", (edge_x, edge_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 3)
    
    cv2.putText(canvas, "Connected Puzzle Shape", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, f"Pieces: {len(artifacts)}", (canvas_width - 150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    _, buffer = cv2.imencode('.png', canvas)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"


@app.route('/compare_multiple_artifacts', methods=['POST'])
@login_required
def compare_multiple_artifacts():
    data = request.get_json()
    artifact_ids = data.get('artifact_ids', [])
    
    if len(artifact_ids) < 2:
        return jsonify({'error': 'Select at least 2 artifacts'}), 400
    
    artifacts = []
    for aid in artifact_ids:
        art = get_artifact_by_id(aid, session['user_id'])
        if not art:
            return jsonify({'error': f'Artifact {aid} not found'}), 404
        artifacts.append(art)
    
    import json
    
    try:
        pair_scores = []
        assembly_images = []
        
        for i in range(len(artifacts)):
            art1 = artifacts[i]
            art2 = artifacts[(i + 1) % len(artifacts)]
            
            result = find_connection(art1['image_path'], art2['image_path'])
            
            puzzle_score = result.get('score', 0)
            connects = result.get('connects', False)
            
            pair_assembly = None
            if result.get('internal'):
                assembly_img = visualise_assembly(result)
                if assembly_img is not None:
                    annotated = mark_connection_on_artifacts(
                        art1['image_path'], art2['image_path'], 
                        result, i + 1
                    )
                    if annotated is not None:
                        _, buffer = cv2.imencode('.png', annotated)
                        pair_assembly = base64.b64encode(buffer).decode('utf-8')
            
            assembly_images.append(pair_assembly)
            
            pair_scores.append({
                'score': float(puzzle_score),
                'can_connect': connects,
                'from_idx': int(i),
                'to_idx': int((i + 1) % len(artifacts)),
                'message': result.get('message', '')
            })
        
        n = len(artifacts)
        avg_puzzle = sum(ps['score'] for ps in pair_scores) / n
        
        final_score = avg_puzzle
        
        disrupting_pieces = []
        if final_score < 40 and len(artifacts) > 2:
            for ps in pair_scores:
                if not ps['can_connect']:
                    disrupting_pieces.append(int(ps['from_idx'] + 1))
        
        connection_points = []
        for i, art in enumerate(artifacts):
            poly = json.loads(art['polygon'])
            poly_arr = np.array(poly)
            
            x_min, y_min = poly_arr.min(axis=0)
            x_max, y_max = poly_arr.max(axis=0)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            
            can_connect_prev = any(ps['to_idx'] == i and ps['can_connect'] for ps in pair_scores)
            can_connect_next = any(ps['from_idx'] == i and ps['can_connect'] for ps in pair_scores)
            
            connection_points.append({
                'artifact_idx': int(i),
                'left_point': {
                    'x': float(x_min),
                    'y': float(cy),
                    'side': 'left',
                    'connects': can_connect_prev
                },
                'right_point': {
                    'x': float(x_max),
                    'y': float(cy),
                    'side': 'right',
                    'connects': can_connect_next
                },
                'top_point': {
                    'x': float(cx),
                    'y': float(y_min),
                    'side': 'top',
                    'connects': False
                },
                'bottom_point': {
                    'x': float(cx),
                    'y': float(y_max),
                    'side': 'bottom',
                    'connects': False
                }
            })
        
        if final_score >= 60:
            verdict = "Can Connect!"
        elif final_score >= 40:
            verdict = "May Connect"
        else:
            if len(artifacts) == 2:
                verdict = "Won't Connect"
            else:
                disrupting_str = ", ".join([f"#{p}" for p in disrupting_pieces])
                verdict = f"Won't Connect - Piece {disrupting_str} disrupts"
        
        return jsonify({
            'artifacts': [
                {'id': a['id'], 'name': a['name'], 'image_base64': a['image_base64']}
                for a in artifacts
            ],
            'assembly_images': assembly_images,
            'scores': {
                'puzzle_fit': float(round(avg_puzzle, 2)),
                'total': float(round(final_score, 2))
            },
            'pair_scores': [
                {
                    'from': int(ps['from_idx']),
                    'to': int(ps['to_idx']),
                    'score': float(ps['score']),
                    'can_connect': bool(ps['can_connect']),
                    'message': ps.get('message', '')
                }
                for ps in pair_scores
            ],
            'connection_points': connection_points,
            'disrupting_pieces': disrupting_pieces,
            'verdict': verdict
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500


@app.route('/clear_database', methods=['POST'])
@login_required
def clear_database():
    password = request.json.get('password') if request.is_json else request.form.get('password')
    
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('SELECT password_hash FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    
    if not check_password(password, user['password_hash']):
        return jsonify({'error': 'Invalid password'}), 401
    
    cursor.execute('SELECT image_path FROM artifacts WHERE user_id = ?', (session['user_id'],))
    for row in cursor.fetchall():
        try:
            os.unlink(row['image_path'])
        except:
            pass
    
    cursor.execute('DELETE FROM artifact_features WHERE artifact_id IN (SELECT id FROM artifacts WHERE user_id = ?)', (session['user_id'],))
    cursor.execute('DELETE FROM artifacts WHERE user_id = ?', (session['user_id'],))
    db.commit()
    
    return jsonify({'message': 'Database cleared successfully'}), 200

if __name__ == '__main__':
    init_db()
    print("Starting The-FlintStones Artifact Comparison Server...")
    print("Upload artifacts to compare them with your collection")
    app.run(host='0.0.0.0', port=5000, debug=True)

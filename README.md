# TheFlintStoneAI - Historical Artifact Classification System

A Flask-based web application that uses computer vision and AI to classify, compare, and manage historical artifact data using YOLO model for object detection and polygon analysis.

## Features

- **Artifact Detection**: Automatically detects artifacts and meters in uploaded images using YOLO
- **Similarity Comparison**: Compares artifacts against a historical database using polygon features
- **Database Management**: Store and manage artifact collections with web interface
- **Password Protection**: Secure database clearing with password authentication (1881)
- **Web Interface**: User-friendly earth-themed interface with real-time analysis
- **Dimension Calculation**: Calculates real-world dimensions using reference meter (default 5cm)

## Project Structure

```
TheFlintStoneAI/
├── main.py                 # Flask web server and API endpoints
├── Segmentation.py         # Computer vision and YOLO processing
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface template
├── static/
│   ├── css/
│   │   └── earth-theme.css # Styling and responsive design
│   ├── js/
│   │   └── earth-theme.js  # Frontend functionality
│   └── images/            # Static images
└── weights/
    └── TheFlintAI.pt      # YOLO model weights (55MB)
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone or download the project to your local machine
2. Navigate to the project directory:
   ```bash
   cd TheFlintStoneAI
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the YOLO model weights file `TheFlintAI.pt` is present in the `weights/` directory

## Usage

### Running the Application

1. Start the Flask server:
   ```bash
   python main.py
   ```

2. The server will automatically start a ngrok tunnel for external access
3. Open the provided ngrok URL in your web browser

### Web Interface

1. **Home**: View artifact statistics and navigate to different sections
2. **Compare**: Upload images to find similar artifacts in the database
3. **Database**: View, search, and manage the artifact collection

### API Endpoints

- `GET /` - Main web interface
- `POST /compare` - Compare uploaded artifact with database
- `POST /add_to_database` - Add new artifact to database
- `GET /database` - View database contents (JSON)
- `POST /clear_database` - Clear database (requires password "1881")

## Technical Details

### Computer Vision Pipeline

1. **Image Processing**: Uses OpenCV for image manipulation
2. **Object Detection**: YOLO model detects artifacts and meters
3. **Polygon Extraction**: Extracts precise polygon coordinates for detected objects
4. **Feature Calculation**: Computes geometric and color features:
   - Area, perimeter, width, height
   - Aspect ratio
   - Mean RGB color values
5. **Similarity Scoring**: Uses Euclidean distance between feature vectors

### Model Configuration

- **Model**: YOLO (Ultralytics framework)
- **Weights**: `TheFlintAI.pt` (55MB)
- **Confidence Threshold**: 0.36
- **Classes**: "Artifact" and "Meter"
- **Reference Meter**: 5cm for scale calibration

### Security Features

- Password-protected database clearing (password: 1881)
- File type validation for image uploads
- Maximum file size limit (16MB)
- Secure filename handling

## Dependencies

- `flask==2.3.3` - Web framework
- `ultralytics==8.0.196` - YOLO model framework
- `numpy==1.24.3` - Numerical computations
- `opencv-python==4.8.1.78` - Computer vision
- `Pillow==10.0.1` - Image processing
- `Werkzeug==2.3.7` - WSGI utilities

## File Upload Support

Supported image formats:
- PNG, JPG, JPEG, GIF, BMP, TIFF

Maximum file size: 16MB

## Development Notes

### Customization

- **Meter Length**: Modify `meter_length_cm` parameter in `get_artifact_polygon_cm()` function
- **Confidence Threshold**: Adjust `conf` variable in `Segmentation.py`
- **Password**: Change password in `main.py` clear_database endpoint
- **Styling**: Modify earth theme colors in `earth-theme.css`

### Error Handling

The application includes comprehensive error handling for:
- Invalid file types
- No artifacts detected
- Processing failures
- Network issues
- Authentication errors

## Performance Considerations

- Images are processed temporarily and cleaned up after analysis
- Database stored in memory (for development - consider persistent database for production)
- Polygon features computed on-demand during comparison
- Real-time processing with loading indicators

## Future Enhancements

- Persistent database storage (PostgreSQL, MongoDB)
- User authentication and authorization
- Batch image processing
- Export functionality for artifact data
- Machine learning model retraining interface
- Advanced similarity algorithms
- Artifact categorization and tagging

## License

This project is part of TheFlintStones historical artifact preservation initiative.

## Support

For issues and questions regarding the artifact classification system, please refer to the project documentation or contact the development team.

---

**Note**: This application is designed for historical artifact research and preservation purposes. Ensure you have appropriate permissions for any artifacts you process.
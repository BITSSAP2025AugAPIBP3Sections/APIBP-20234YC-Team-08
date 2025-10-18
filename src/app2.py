import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, label, maximum_filter

# Limit TensorFlow threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

MODEL_PATH = Path("model/mnist_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

class CharacterSegmenter:
    def __init__(self, min_char_width=8, min_char_height=15, max_char_width=100, max_char_height=150):
        self.min_char_width = min_char_width
        self.min_char_height = min_char_height
        self.max_char_width = max_char_width
        self.max_char_height = max_char_height
    
    def preprocess_image(self, image):
        """Clean and prepare image for segmentation"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv2.medianBlur(gray, 3)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def find_text_lines(self, binary_image):
        """Find text lines using horizontal projection"""
        # Horizontal projection
        h_projection = np.sum(binary_image, axis=1)
        
        # Find line boundaries
        line_threshold = np.max(h_projection) * 0.1 if np.max(h_projection) > 0 else 0
        in_line = h_projection > line_threshold
        
        lines = []
        start = None
        
        for i, is_text in enumerate(in_line):
            if is_text and start is None:
                start = i
            elif not is_text and start is not None:
                if i - start > self.min_char_height:  # Minimum line height
                    lines.append((start, i))
                start = None
        
        # Handle case where line goes to end of image
        if start is not None:
            lines.append((start, len(in_line)))
        
        # If no lines found, use entire image
        if not lines:
            lines = [(0, binary_image.shape[0])]
        
        return lines
    
    def segment_line_by_connected_components(self, line_image):
        """Segment a text line into characters using connected components"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_image, connectivity=8)
        
        character_boxes = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            
            # Filter by size
            if (self.min_char_width <= w <= self.max_char_width and 
                self.min_char_height <= h <= self.max_char_height and
                area > 20):  # Minimum area threshold
                
                character_boxes.append((x, y, w, h, area))
        
        return character_boxes
    
    def segment_line_by_projection(self, line_image):
        """Segment using vertical projection - useful for touching characters"""
        # Vertical projection
        v_projection = np.sum(line_image, axis=0)
        
        # Smooth the projection
        smoothed = ndimage.gaussian_filter1d(v_projection, sigma=1)
        
        # Find valleys (potential split points)
        threshold = np.max(smoothed) * 0.3 if np.max(smoothed) > 0 else 0
        in_char = smoothed > threshold
        
        segments = []
        start = None
        
        for i, is_text in enumerate(in_char):
            if is_text and start is None:
                start = i
            elif not is_text and start is not None:
                if i - start > self.min_char_width:
                    segments.append((start, i))
                start = None
        
        if start is not None:
            segments.append((start, len(in_char)))
        
        # Convert to bounding boxes
        character_boxes = []
        h, w = line_image.shape
        
        for start_x, end_x in segments:
            # Find actual y boundaries within this segment
            segment = line_image[:, start_x:end_x]
            if np.sum(segment) > 0:  # Has content
                y_coords = np.where(np.sum(segment, axis=1) > 0)[0]
                if len(y_coords) > 0:
                    y_start, y_end = y_coords[0], y_coords[-1] + 1
                    character_boxes.append((start_x, y_start, end_x - start_x, y_end - y_start, np.sum(segment)))
        
        return character_boxes
    
    def segment_image(self, image, method='hybrid'):
        """Main segmentation function"""
        # Preprocess
        binary = self.preprocess_image(image)
        
        # Find text lines
        lines = self.find_text_lines(binary)
        
        all_characters = []
        
        for line_start, line_end in lines:
            line_image = binary[line_start:line_end, :]
            
            if method == 'connected_components':
                char_boxes = self.segment_line_by_connected_components(line_image)
            elif method == 'projection':
                char_boxes = self.segment_line_by_projection(line_image)
            else:  # hybrid
                # Try connected components first
                char_boxes = self.segment_line_by_connected_components(line_image)
                
                # If too few characters found, try projection
                if len(char_boxes) < 2:
                    proj_boxes = self.segment_line_by_projection(line_image)
                    if len(proj_boxes) > len(char_boxes):
                        char_boxes = proj_boxes
            
            # Convert to global coordinates
            global_boxes = []
            for x, y, w, h, area in char_boxes:
                global_boxes.append((x, y + line_start, w, h, area))
            
            all_characters.extend(global_boxes)
        
        # Sort all characters by position (top to bottom, left to right)
        all_characters.sort(key=lambda box: (box[1], box[0]))
        
        return binary, all_characters

def extract_character_image(binary_image, x, y, w, h, target_size=(28, 28)):
    """Extract and preprocess character for model prediction"""
    # Extract character region
    char_region = binary_image[y:y+h, x:x+w]
    
    # Add padding
    pad = max(w, h) // 4
    padded_h, padded_w = h + 2*pad, w + 2*pad
    padded_image = np.zeros((padded_h, padded_w), dtype=np.uint8)
    
    # Center the character
    start_y = (padded_h - h) // 2
    start_x = (padded_w - w) // 2
    padded_image[start_y:start_y+h, start_x:start_x+w] = char_region
    
    # Convert to PIL and resize
    pil_image = Image.fromarray(padded_image)
    resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Normalize
    char_array = np.array(resized, dtype=np.float32) / 255.0
    
    return char_array

def predict_characters(model, binary_image, character_boxes):
    """Predict all characters using the ML model"""
    predictions = []
    processed_chars = []
    
    for x, y, w, h, area in character_boxes:
        # Extract and preprocess
        char_array = extract_character_image(binary_image, x, y, w, h)
        processed_chars.append(char_array)
        
        # Predict
        char_input = char_array[np.newaxis, ..., np.newaxis]
        pred = model.predict(char_input, verbose=0)
        predicted_digit = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]))
        
        predictions.append({
            'digit': predicted_digit,
            'confidence': confidence,
            'bbox': (x, y, w, h),
            'area': area
        })
    
    return predictions, processed_chars

def visualize_segmentation(image, binary_image, character_boxes, predictions=None):
    """Visualize the segmentation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Binary image
    axes[0, 1].imshow(binary_image, cmap='gray')
    axes[0, 1].set_title('Preprocessed Binary Image')
    axes[0, 1].axis('off')
    
    # Segmentation results
    axes[1, 0].imshow(image, cmap='gray')
    for i, (x, y, w, h, area) in enumerate(character_boxes):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='red', facecolor='none')
        axes[1, 0].add_patch(rect)
        axes[1, 0].text(x, y-2, str(i), color='red', fontsize=10)
    axes[1, 0].set_title(f'Detected Character Regions ({len(character_boxes)} found)')
    axes[1, 0].axis('off')
    
    # Predictions
    if predictions:
        axes[1, 1].imshow(image, cmap='gray')
        for pred in predictions:
            x, y, w, h = pred['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='blue', facecolor='none')
            axes[1, 1].add_patch(rect)
            axes[1, 1].text(x, y-2, f"{pred['digit']}", color='blue', 
                           fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Predictions')
        axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

# Streamlit App
st.title("Bank Cheque Character Recognition")
st.write("Upload a bank cheque image to detect and recognize handwritten digits")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload Image", "Draw Digit (Original)"])

with tab1:
    uploaded_file = st.file_uploader("Choose a cheque image...", 
                                   type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'])
    
    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        st.image(image, caption="Original Cheque Image", use_column_width=True)
        
        # Create segmenter instance with sidebar controls
        st.sidebar.header("Segmentation Parameters")
        min_char_width = st.sidebar.slider("Minimum Character Width", 5, 30, 8)
        min_char_height = st.sidebar.slider("Minimum Character Height", 10, 50, 15)
        max_char_width = st.sidebar.slider("Maximum Character Width", 50, 200, 100)
        max_char_height = st.sidebar.slider("Maximum Character Height", 80, 300, 150)
        
        segmentation_method = st.sidebar.selectbox(
            "Segmentation Method",
            ["hybrid", "connected_components", "projection"]
        )
        
        segmenter = CharacterSegmenter(
            min_char_width=min_char_width,
            min_char_height=min_char_height,
            max_char_width=max_char_width,
            max_char_height=max_char_height
        )
        
        if st.button("Detect and Recognize Characters"):
            with st.spinner("Processing image..."):
                # Segment the image
                binary_image, character_boxes = segmenter.segment_image(
                    image_array, method=segmentation_method
                )
                
                if character_boxes:
                    st.success(f"Found {len(character_boxes)} character regions")
                    
                    # Predict characters
                    predictions, processed_chars = predict_characters(
                        model, binary_image, character_boxes
                    )
                    
                    # Visualize results
                    fig = visualize_segmentation(
                        image_array, binary_image, character_boxes, predictions
                    )
                    st.pyplot(fig)
                    
                    # Show individual character predictions
                    st.subheader("Individual Character Predictions")
                    
                    # Create columns for character display
                    num_cols = min(len(predictions), 6)
                    if num_cols > 0:
                        cols = st.columns(num_cols)
                        
                        for i, (pred, char_img) in enumerate(zip(predictions, processed_chars)):
                            with cols[i % num_cols]:
                                st.image(char_img, caption=f"Predicted: {pred['digit']}\n"
                                       f"Confidence: {pred['confidence']:.3f}", 
                                       width=80)
                    
                    # Show extracted number sequence
                    extracted_numbers = [str(pred['digit']) for pred in predictions]
                    st.subheader("Extracted Character Sequence")
                    st.write("**Detected characters (left to right):**", " ".join(extracted_numbers))
                    st.write("**As number:**", "".join(extracted_numbers))
                    
                    # Show confidence scores
                    avg_confidence = np.mean([pred['confidence'] for pred in predictions])
                    st.write(f"**Average Confidence:** {avg_confidence:.3f}")
                    
                    # Show low confidence warnings
                    low_conf_chars = [pred for pred in predictions if pred['confidence'] < 0.5]
                    if low_conf_chars:
                        st.warning(f"⚠️ {len(low_conf_chars)} characters have low confidence (<0.5)")
                    
                else:
                    st.warning("No character regions detected. Try adjusting the parameters in the sidebar.")


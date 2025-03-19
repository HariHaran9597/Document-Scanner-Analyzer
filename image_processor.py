import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self):
        # Default preprocessing parameters
        self.params = {
            'blur_kernel_width': 3,  # Reduced blur for better detail
            'blur_kernel_height': 3,
            'threshold_block_size': 15,  # Increased block size
            'threshold_constant': 8,  # Adjusted for better contrast
            'contour_approx_epsilon': 0.02,
            'sharpen_kernel': 0.5  # New parameter for sharpening
        }

    def set_parameters(self, **kwargs):
        """Update preprocessing parameters"""
        self.params.update(kwargs)

    def preprocess_image(self, image):
        """Preprocess image with configurable parameters"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create sharpening kernel
        sharpen_kernel = np.array([
            [-1,-1,-1],
            [-1, 9,-1],
            [-1,-1,-1]
        ]) * self.params['sharpen_kernel']
        
        # Apply sharpening
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        
        # Create kernel tuple from width and height
        kernel_size = (
            self.params['blur_kernel_width'] if self.params['blur_kernel_width'] % 2 == 1 else self.params['blur_kernel_width'] + 1,
            self.params['blur_kernel_height'] if self.params['blur_kernel_height'] % 2 == 1 else self.params['blur_kernel_height'] + 1
        )
        
        # Apply light Gaussian blur
        blurred = cv2.GaussianBlur(sharpened, kernel_size, 0)
        
        # Apply adaptive thresholding with improved parameters
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.params['threshold_block_size'],
            self.params['threshold_constant']
        )
        
        # Denoise the image
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised

    def find_document_contour(self, image):
        """Find document contour with visual feedback"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour
        peri = cv2.arcLength(largest_contour, True)
        epsilon = self.params['contour_approx_epsilon'] * peri
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            return approx, largest_contour
        return None, largest_contour

    @staticmethod
    def order_points(pts):
        # Order points in: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        return rect

    @staticmethod
    def perspective_transform(image, pts):
        rect = ImageProcessor.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute width
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Compute height
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Create destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Calculate perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def draw_contours(self, image, contour, approx=None):
        """Draw detected contours on the image"""
        result = image.copy()
        
        # Draw the main contour
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        
        # If we have the approximated rectangle, draw its corners
        if approx is not None:
            for point in approx.reshape(-1, 2):
                cv2.circle(result, tuple(point), 5, (0, 0, 255), -1)
        
        return result

    def draw_text_regions(self, image, regions, colors=None):
        """Draw text regions with their confidence scores"""
        result = image.copy()
        
        if colors is None:
            colors = {
                90: (0, 255, 0),    # High confidence (green)
                70: (0, 255, 255),  # Medium confidence (yellow)
                0: (0, 0, 255)      # Low confidence (red)
            }
        
        for region in regions:
            # Get color based on confidence
            color = None
            conf = region['conf']
            for threshold, col in colors.items():
                if conf >= threshold:
                    color = col
                    break
            if color is None:
                color = colors[0]  # Default to lowest confidence color

            # Draw rectangle around text region
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Add confidence score
            conf_text = f"{conf:.1f}%"
            cv2.putText(result, conf_text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        return result

    def process_document(self, image_input, draw_bounds=False, draw_text=False):
        """Process document with optional visual feedback"""
        # Handle both file path strings and image objects
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError("Could not read image from path")
        else:
            image = image_input

        # Make a copy for visualization
        original = image.copy()
        
        # Preprocess
        processed = self.preprocess_image(image.copy())
        
        # Find document contour
        contour, full_contour = self.find_document_contour(processed)
        
        if contour is None:
            warped = image
        else:
            # Reshape contour and apply perspective transform
            pts = contour.reshape(4, 2).astype("float32")
            warped = self.perspective_transform(image, pts)
        
        # Prepare visualization results
        results = {
            'processed': warped
        }
        
        if draw_bounds:
            results['annotated'] = self.draw_contours(
                original, 
                full_contour, 
                contour
            )
        
        return results

    def batch_process(self, images):
        """Process multiple documents in batch"""
        results = []
        for img in images:
            try:
                processed = self.process_document(img)
                results.append({
                    'success': True,
                    'image': processed
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e)
                })
        return results
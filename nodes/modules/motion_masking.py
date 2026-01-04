
import cv2
import numpy as np
import logging

class MotionMasking:
    """
    High-End Motion Segmentation for Adaptive 4D Gaussian Splatting.
    
    Generates a weight mask based on the difference between the current frame
    and the previous frame. This allows the trainer to:
    1. Lock/Stabilize static regions (High weight on low diff)
    2. Focus densification on moving regions (High weight on high diff)
    """
    
    def __init__(self, sensitivity=0.5, dilation=5):
        self.sensitivity = sensitivity
        self.dilation = dilation
        self.logger = logging.getLogger("FreeFlow.MotionMasking")
        
    def compute_mask(self, current_img_path, prev_img_path, output_path, method="Optical Flow (Robust)"):
        """
        Compute difference mask between two images.
        Methods:
        - "Optical Flow (Robust)": Uses Farneback flow magnitude. Best for stability.
        - "Simple Diff (Fast)": Uses abs(img1-img2). Prone to flicker.
        """
        try:
            # Read images
            img1 = cv2.imread(str(current_img_path))
            img2 = cv2.imread(str(prev_img_path))
            
            if img1 is None or img2 is None:
                self.logger.warning("Could not read images for masking. Skipping.")
                return None
                
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            mask = None
            
            if method and "Simple" in method:
                # --- SIMPLE DIFF (Legacy) ---
                diff = cv2.absdiff(gray1, gray2)
                
                # Threshold
                thresh_val = int((1.0 - self.sensitivity) * 50) + 10 
                _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
                
                # Cleanup
                kernel = np.ones((self.dilation, self.dilation), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=2)
                mask = cv2.erode(mask, kernel, iterations=1)
                
            else:
                # --- OPTICAL FLOW (Default/Robust) ---
                # Parameters: pyr=0.5, levels=3, winsize=15, iter=3, poly=5, sig=1.2
                flow = cv2.calcOpticalFlowFarneback(gray2, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
                # Compute Magnitude
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Thresholding
                base_threshold = 2.0 * (1.1 - self.sensitivity) 
                _, mask = cv2.threshold(mag, base_threshold, 255, cv2.THRESH_BINARY)
                mask = mask.astype(np.uint8)
                
                # Refinement
                kernel = np.ones((self.dilation, self.dilation), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Save
            cv2.imwrite(str(output_path), mask)
            return output_path
            
        except Exception as e:
            self.logger.error(f"Motion Masking Failed: {e}")
            return None
            


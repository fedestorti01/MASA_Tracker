import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import json

class DualImageMatchingApp:
    def __init__(self):
        # Initialize variables
        self.image1 = None
        self.image2 = None
        self.image1_path = ""
        self.image2_path = ""
        self.display_image1 = None
        self.display_image2 = None
        
        # Matching points storage
        self.points1 = []  # Points on image1
        self.points2 = []  # Points on image2
        self.point_pairs = []  # List of (point1, point2) pairs
        self.temp_point1 = None  # Temporary storage for first point selection
        self.adding_points_mode = False
        
        # Homography variables
        self.homography_matrix = None
        self.homography_mode = False
        
        # Display parameters
        self.window_width = 1920
        self.window_height = 1080
        self.button_height = 80
        self.image_height = self.window_height - self.button_height
        self.half_width = self.window_width // 2
        
        # Colors for visualization
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                      (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0)]
        
        # Window names
        self.window_name = "Dual Image Matching - Click points to create pairs"
        
        self.setup_display()
        
    def setup_display(self):
        """Initialize the display window"""
        # Create main display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        
        # Create initial display
        self.update_display()
        
        # Set mouse callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def create_display_image(self):
        """Create the main display image with both panels and buttons"""
        # Create main canvas
        canvas = np.ones((self.window_height, self.window_width, 3), dtype=np.uint8) * 240
        
        # Draw dividing line
        cv2.line(canvas, (self.half_width, 0), (self.half_width, self.image_height), (0, 0, 0), 2)
        
        # Add images if loaded
        if self.display_image1 is not None:
            # Resize image1 to fit left half
            resized_img1 = self.resize_image_to_fit(self.display_image1, self.half_width, self.image_height)
            h1, w1 = resized_img1.shape[:2]
            
            # Center the image in the left half
            y_offset = (self.image_height - h1) // 2
            x_offset = (self.half_width - w1) // 2
            
            canvas[y_offset:y_offset+h1, x_offset:x_offset+w1] = resized_img1
            self.image1_offset = (x_offset, y_offset)
            self.image1_scale = min(self.half_width / self.image1.shape[1], self.image_height / self.image1.shape[0])
        else:
            # Draw placeholder for image1
            cv2.putText(canvas, "Image 1", (self.half_width//4, self.image_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 3)
            cv2.putText(canvas, "Click 'Load Image 1' to load", (50, self.image_height//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            self.image1_offset = (0, 0)
            self.image1_scale = 1.0
            
        if self.display_image2 is not None:
            # Resize image2 to fit right half
            resized_img2 = self.resize_image_to_fit(self.display_image2, self.half_width, self.image_height)
            h2, w2 = resized_img2.shape[:2]
            
            # Center the image in the right half
            y_offset = (self.image_height - h2) // 2
            x_offset = self.half_width + (self.half_width - w2) // 2
            
            canvas[y_offset:y_offset+h2, x_offset:x_offset+w2] = resized_img2
            self.image2_offset = (x_offset, y_offset)
            self.image2_scale = min(self.half_width / self.image2.shape[1], self.image_height / self.image2.shape[0])
        else:
            # Draw placeholder for image2
            cv2.putText(canvas, "Image 2", (self.half_width + self.half_width//4, self.image_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 3)
            cv2.putText(canvas, "Click 'Load Image 2' to load", (self.half_width + 50, self.image_height//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            self.image2_offset = (self.half_width, 0)
            self.image2_scale = 1.0
        
        # Draw buttons at the bottom
        self.draw_buttons(canvas)
        
        return canvas
    
    def resize_image_to_fit(self, image, max_width, max_height):
        """Resize image to fit within given dimensions while maintaining aspect ratio"""
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        return cv2.resize(image, (new_width, new_height))
    
    def draw_buttons(self, canvas):
        """Draw control buttons at the bottom of the canvas"""
        button_width = self.window_width // 7
        button_y = self.image_height
        
        # Button colors and texts
        buttons = [
            ("Load Image 1", (180, 180, 255)),
            ("Load Image 2", (180, 255, 180)),
            ("Add Points" if not self.adding_points_mode else "Stop Adding", (255, 255, 180) if not self.adding_points_mode else (255, 180, 100)),
            ("Load Points", (200, 180, 255)),
            ("Delete Point", (255, 200, 200)),
            ("Compute Homography", (180, 255, 255) if not self.homography_mode else (100, 200, 200)),
            ("Export Points", (255, 180, 180))
        ]
        
        for i, (text, color) in enumerate(buttons):
            x_start = i * button_width
            x_end = (i + 1) * button_width
            
            # Draw button background
            cv2.rectangle(canvas, (x_start, button_y), (x_end, self.window_height), color, -1)
            cv2.rectangle(canvas, (x_start, button_y), (x_end, self.window_height), (0, 0, 0), 2)
            
            # Draw button text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = x_start + (button_width - text_size[0]) // 2
            text_y = button_y + (self.button_height + text_size[1]) // 2
            cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Store button positions for click detection
        self.button_regions = [
            (0, button_y, button_width, self.window_height),
            (button_width, button_y, 2*button_width, self.window_height),
            (2*button_width, button_y, 3*button_width, self.window_height),
            (3*button_width, button_y, 4*button_width, self.window_height),
            (4*button_width, button_y, 5*button_width, self.window_height),
            (5*button_width, button_y, 6*button_width, self.window_height),
            (6*button_width, button_y, 7*button_width, self.window_height)
        ]
    
    def draw_points_on_display_images(self):
        """Draw points and pairs on the display images"""
        if self.image1 is not None:
            self.display_image1 = self.image1.copy()
            
        if self.image2 is not None:
            self.display_image2 = self.image2.copy()
        
        # Draw existing point pairs
        for i, (p1, p2) in enumerate(self.point_pairs):
            color = self.colors[i % len(self.colors)]
            
            if self.display_image1 is not None:
                cv2.circle(self.display_image1, p1, 12, color, -1)
                cv2.circle(self.display_image1, p1, 15, (255, 255, 255), 3)
                cv2.putText(self.display_image1, str(i+1), 
                           (p1[0] + 20, p1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
            if self.display_image2 is not None:
                cv2.circle(self.display_image2, p2, 12, color, -1)
                cv2.circle(self.display_image2, p2, 15, (255, 255, 255), 3)
                cv2.putText(self.display_image2, str(i+1), 
                           (p2[0] + 20, p2[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Draw temporary point if exists
        if self.temp_point1 is not None and self.display_image1 is not None:
            cv2.circle(self.display_image1, self.temp_point1, 12, (0, 255, 255), -1)
            cv2.circle(self.display_image1, self.temp_point1, 15, (255, 255, 255), 3)
            cv2.putText(self.display_image1, "?", 
                       (self.temp_point1[0] + 20, self.temp_point1[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    def update_display(self):
        """Update the main display"""
        self.draw_points_on_display_images()
        display = self.create_display_image()
        cv2.imshow(self.window_name, display)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks and movements"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is in button area
            if y >= self.image_height:
                self.handle_button_click(x, y)
            elif self.adding_points_mode:
                self.handle_image_click(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.homography_mode:
            # Handle mouse movement for homography visualization
            self.handle_mouse_move(x, y)
    
    def handle_button_click(self, x, y):
        """Handle clicks on buttons"""
        button_width = self.window_width // 7
        button_index = x // button_width
        
        if button_index == 0:  # Load Image 1
            self.load_image1()
        elif button_index == 1:  # Load Image 2
            self.load_image2()
        elif button_index == 2:  # Add/Stop Points
            self.toggle_point_mode()
        elif button_index == 3:  # Load Points
            self.load_points()
        elif button_index == 4:  # Delete Point
            self.delete_point()
        elif button_index == 5:  # Compute Homography
            self.compute_homography()
        elif button_index == 6:  # Export Points
            self.export_points()
    
    def handle_image_click(self, x, y):
        """Handle clicks on image areas"""
        if x < self.half_width:  # Left side - Image 1
            if self.image1 is not None:
                # Convert screen coordinates to image coordinates
                img_x, img_y = self.screen_to_image_coords(x, y, 1)
                if 0 <= img_x < self.image1.shape[1] and 0 <= img_y < self.image1.shape[0]:
                    if self.temp_point1 is None:
                        self.temp_point1 = (int(img_x), int(img_y))
                        print(f"Point selected in Image 1: ({int(img_x)}, {int(img_y)}). Now click corresponding point in Image 2.")
                    else:
                        self.temp_point1 = (int(img_x), int(img_y))
                        print(f"Point updated in Image 1: ({int(img_x)}, {int(img_y)}). Now click corresponding point in Image 2.")
                    self.update_display()
        else:  # Right side - Image 2
            if self.image2 is not None and self.temp_point1 is not None:
                # Convert screen coordinates to image coordinates
                img_x, img_y = self.screen_to_image_coords(x, y, 2)
                if 0 <= img_x < self.image2.shape[1] and 0 <= img_y < self.image2.shape[0]:
                    point2 = (int(img_x), int(img_y))
                    self.point_pairs.append((self.temp_point1, point2))
                    self.points1.append(self.temp_point1)
                    self.points2.append(point2)
                    
                    print(f"Point pair {len(self.point_pairs)} created: Image1{self.temp_point1} <-> Image2{point2}")
                    self.temp_point1 = None
                    self.update_display()
    
    def screen_to_image_coords(self, screen_x, screen_y, image_num):
        """Convert screen coordinates to image coordinates"""
        if image_num == 1:
            offset = self.image1_offset
            scale = self.image1_scale
        else:
            offset = self.image2_offset
            scale = self.image2_scale
            
        # Convert screen coordinates to image coordinates
        image_x = (screen_x - offset[0]) / scale
        image_y = (screen_y - offset[1]) / scale
        
        return image_x, image_y
    
    def load_image1(self):
        """Load first image using tkinter file dialog"""
        # Create a temporary tkinter root for file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        file_path = filedialog.askopenfilename(
            title="Select Image 1",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        root.destroy()  # Clean up the temporary root
        
        if file_path:
            try:
                self.image1 = cv2.imread(file_path)
                if self.image1 is None:
                    raise ValueError("Could not load image")
                
                self.image1_path = file_path
                self.display_image1 = self.image1.copy()
                self.update_display()
                
                filename = os.path.basename(file_path)
                print(f"Loaded Image 1: {filename} ({self.image1.shape[1]}x{self.image1.shape[0]})")
                
            except Exception as e:
                print(f"Failed to load image 1: {str(e)}")
    
    def load_image2(self):
        """Load second image using tkinter file dialog"""
        # Create a temporary tkinter root for file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        file_path = filedialog.askopenfilename(
            title="Select Image 2",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        
        root.destroy()  # Clean up the temporary root
        
        if file_path:
            try:
                self.image2 = cv2.imread(file_path)
                if self.image2 is None:
                    raise ValueError("Could not load image")
                
                self.image2_path = file_path
                self.display_image2 = self.image2.copy()
                self.update_display()
                
                filename = os.path.basename(file_path)
                print(f"Loaded Image 2: {filename} ({self.image2.shape[1]}x{self.image2.shape[0]})")
                
            except Exception as e:
                print(f"Failed to load image 2: {str(e)}")
    
    def toggle_point_mode(self):
        """Toggle point adding mode"""
        if self.image1 is not None and self.image2 is not None:
            self.adding_points_mode = not self.adding_points_mode
            if self.adding_points_mode:
                print("Point adding mode ON - Click on corresponding points in both images")
                cv2.setWindowTitle(self.window_name, "Point Adding Mode ON - Click corresponding points")
            else:
                print(f"Point adding mode OFF - {len(self.point_pairs)} pairs created")
                cv2.setWindowTitle(self.window_name, f"Point Adding Mode OFF - {len(self.point_pairs)} pairs created")
                self.temp_point1 = None
            self.update_display()
        else:
            print("Please load both images first!")
    
    def delete_point(self):
        """Delete a specific point pair by number"""
        if len(self.point_pairs) == 0:
            print("No points to delete!")
            return
        
        # Create a temporary tkinter root for input dialog
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Show current points
        points_info = f"Current point pairs ({len(self.point_pairs)} total):\n"
        for i, (p1, p2) in enumerate(self.point_pairs):
            points_info += f"  {i+1}: Image1{p1} <-> Image2{p2}\n"
        
        print(points_info)
        
        # Ask user which point to delete
        from tkinter import simpledialog
        point_number = simpledialog.askinteger(
            "Delete Point",
            f"Enter point number to delete (1-{len(self.point_pairs)}):\n\n{points_info}",
            minvalue=1,
            maxvalue=len(self.point_pairs)
        )
        
        root.destroy()  # Clean up the temporary root
        
        if point_number is not None:
            try:
                # Convert to 0-based index
                index = point_number - 1
                
                # Get the point pair to be deleted
                deleted_pair = self.point_pairs[index]
                deleted_p1 = self.points1[index]
                deleted_p2 = self.points2[index]
                
                # Remove the point pair
                self.point_pairs.pop(index)
                self.points1.pop(index)
                self.points2.pop(index)
                
                # Update display
                self.update_display()
                
                print(f"Deleted point pair {point_number}: Image1{deleted_p1} <-> Image2{deleted_p2}")
                print(f"Remaining points: {len(self.point_pairs)}")
                
                # Show updated list if there are remaining points
                if len(self.point_pairs) > 0:
                    print("Updated point pairs:")
                    for i, (p1, p2) in enumerate(self.point_pairs):
                        print(f"  {i+1}: Image1{p1} <-> Image2{p2}")
                else:
                    print("No point pairs remaining")
                
            except Exception as e:
                print(f"Failed to delete point: {str(e)}")
    
    def compute_homography(self):
        """Compute homography matrix between the two sets of points"""
        if len(self.point_pairs) < 4:
            print("Need at least 4 point pairs to compute homography!")
            return
        
        try:
            # Prepare points for homography calculation
            src_points = np.array(self.points1, dtype=np.float32)
            dst_points = np.array(self.points2, dtype=np.float32)
            
            # Calculate homography using different methods
            H, mask = cv2.findHomography(src_points, dst_points)
            H_LMEDS, mask_LMEDS = cv2.findHomography(src_points, dst_points, cv2.LMEDS)
            H_RANSAC, mask_RANSAC = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
            
            # Project points to evaluate quality
            projected_points = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H)
            projected_points_LMEDS = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H_LMEDS)
            projected_points_RANSAC = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H_RANSAC)
            
            # Calculate distances to evaluate which method is best
            distances = np.linalg.norm(projected_points.reshape(-1, 2) - dst_points, axis=1)
            distances_LMEDS = np.linalg.norm(projected_points_LMEDS.reshape(-1, 2) - dst_points, axis=1)
            distances_RANSAC = np.linalg.norm(projected_points_RANSAC.reshape(-1, 2) - dst_points, axis=1)
            
            # Choose the best method
            algorithm = "using all point pairs"
            self.homography_matrix = H
            if np.mean(distances_LMEDS) < np.mean(distances):
                algorithm = "least-Median robust method"
                self.homography_matrix = H_LMEDS
                distances = distances_LMEDS
            if np.mean(distances_RANSAC) < np.mean(distances):
                algorithm = "RANSAC-based robust method"
                self.homography_matrix = H_RANSAC
                distances = distances_RANSAC
            
            # Enable homography mode
            self.homography_mode = True
            self.adding_points_mode = False  # Disable point adding mode
            
            # Update display
            self.update_display()
            
            # Print results
            print(f"Homography computed successfully using {algorithm}")
            print(f"Matrix shape: {self.homography_matrix.shape}")
            print(f"Reprojection errors:")
            print(f"  Max distance: {np.max(distances):.2f} pixels")
            print(f"  Min distance: {np.min(distances):.2f} pixels")
            print(f"  Mean distance: {np.mean(distances):.2f} pixels")
            print(f"  Variance: {np.var(distances):.2f}")
            print("Homography mode ON - Move mouse over Image 1 to see projected points in Image 2")
            
            # Update window title
            cv2.setWindowTitle(self.window_name, "Homography Mode ON - Move mouse over Image 1")
            
        except Exception as e:
            print(f"Failed to compute homography: {str(e)}")
            self.homography_matrix = None
            self.homography_mode = False
    
    def handle_mouse_move(self, x, y):
        """Handle mouse movement for homography visualization"""
        if not self.homography_mode or self.homography_matrix is None:
            return
        
        # Check if mouse is over the first image
        if x < self.half_width and y < self.image_height:
            if self.image1 is not None:
                # Convert screen coordinates to image coordinates
                img_x, img_y = self.screen_to_image_coords(x, y, 1)
                
                # Check if point is within image bounds
                if 0 <= img_x < self.image1.shape[1] and 0 <= img_y < self.image1.shape[0]:
                    # Project point using homography
                    point = np.array([[[img_x, img_y]]], dtype=np.float32)
                    projected_point = cv2.perspectiveTransform(point, self.homography_matrix)
                    
                    px, py = projected_point[0][0]
                    px, py = int(round(px)), int(round(py))
                    
                    # Create copies of the display images for real-time visualization
                    if self.display_image1 is not None and self.display_image2 is not None:
                        img1_copy = self.display_image1.copy()
                        img2_copy = self.display_image2.copy()
                        
                        # Draw current mouse position on image1
                        cv2.circle(img1_copy, (int(img_x), int(img_y)), 7, (0, 0, 255), 2)
                        cv2.putText(img1_copy, f"({int(img_x)}, {int(img_y)})", 
                                   (int(img_x) + 10, int(img_y) - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        
                        # Draw projected point on image2 if it's within bounds
                        if 0 <= px < self.image2.shape[1] and 0 <= py < self.image2.shape[0]:
                            cv2.circle(img2_copy, (px, py), 5, (0, 0, 255), 2)
                            cv2.putText(img2_copy, f"({px}, {py})", 
                                       (px + 10, py - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        
                        # Create and show the combined display
                        canvas = self.create_display_canvas_with_images(img1_copy, img2_copy)
                        cv2.imshow(self.window_name, canvas)
    
    def create_display_canvas_with_images(self, img1, img2):
        """Create display canvas with custom images (for real-time homography visualization)"""
        # Create main canvas
        canvas = np.ones((self.window_height, self.window_width, 3), dtype=np.uint8) * 240
        
        # Draw dividing line
        cv2.line(canvas, (self.half_width, 0), (self.half_width, self.image_height), (0, 0, 0), 2)
        
        # Add image1 if provided
        if img1 is not None:
            resized_img1 = self.resize_image_to_fit(img1, self.half_width, self.image_height)
            h1, w1 = resized_img1.shape[:2]
            y_offset = (self.image_height - h1) // 2
            x_offset = (self.half_width - w1) // 2
            canvas[y_offset:y_offset+h1, x_offset:x_offset+w1] = resized_img1
        
        # Add image2 if provided
        if img2 is not None:
            resized_img2 = self.resize_image_to_fit(img2, self.half_width, self.image_height)
            h2, w2 = resized_img2.shape[:2]
            y_offset = (self.image_height - h2) // 2
            x_offset = self.half_width + (self.half_width - w2) // 2
            canvas[y_offset:y_offset+h2, x_offset:x_offset+w2] = resized_img2
        
        # Draw buttons
        self.draw_buttons(canvas)
        
        return canvas
    
    def load_points(self):
        """Load previously saved matching points from a file"""
        # Create a temporary tkinter root for file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        file_path = filedialog.askopenfilename(
            title="Load Matching Points",
            filetypes=[("JSON files", "*.json"), ("YAML files", "*.yaml"), ("Text files", "*.txt")]
        )
        
        root.destroy()  # Clean up the temporary root
        
        if file_path:
            try:
                # Clear existing points and homography
                self.points1 = []
                self.points2 = []
                self.point_pairs = []
                self.temp_point1 = None
                self.homography_matrix = None
                self.homography_mode = False
                
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                elif file_path.endswith('.yaml'):
                    import yaml
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                else:
                    # For text files, we'll provide basic parsing
                    print("Text file loading not fully supported. Please use JSON or YAML files for loading points.")
                    return
                
                # Extract points from loaded data
                if 'matching_points' in data:
                    matching_data = data['matching_points']
                    
                    if 'point_pairs' in matching_data:
                        # Load point pairs
                        for pair in matching_data['point_pairs']:
                            p1 = tuple(pair[0])
                            p2 = tuple(pair[1])
                            self.point_pairs.append((p1, p2))
                            self.points1.append(p1)
                            self.points2.append(p2)
                    
                    elif 'points1' in matching_data and 'points2' in matching_data:
                        # Load separate point lists
                        points1_list = matching_data['points1']
                        points2_list = matching_data['points2']
                        
                        if len(points1_list) == len(points2_list):
                            for p1, p2 in zip(points1_list, points2_list):
                                p1_tuple = tuple(p1)
                                p2_tuple = tuple(p2)
                                self.point_pairs.append((p1_tuple, p2_tuple))
                                self.points1.append(p1_tuple)
                                self.points2.append(p2_tuple)
                        else:
                            print("Warning: Mismatch in number of points between images")
                
                # Load homography matrix if available
                if 'homography' in data and data['homography']['computed']:
                    try:
                        matrix_list = data['homography']['matrix']
                        if matrix_list:
                            self.homography_matrix = np.array(matrix_list).reshape(3, 3)
                            self.homography_mode = True
                            print("Loaded homography matrix")
                            print("Homography mode ON - Move mouse over Image 1 to see projected points")
                    except Exception as e:
                        print(f"Warning: Could not load homography matrix: {e}")
                        self.homography_matrix = None
                        self.homography_mode = False
                
                # Load image paths if available and images are not already loaded
                if 'image1_path' in data and self.image1 is None:
                    img1_path = data['image1_path']
                    if os.path.exists(img1_path):
                        self.image1 = cv2.imread(img1_path)
                        if self.image1 is not None:
                            self.image1_path = img1_path
                            self.display_image1 = self.image1.copy()
                            print(f"Auto-loaded Image 1: {os.path.basename(img1_path)}")
                
                if 'image2_path' in data and self.image2 is None:
                    img2_path = data['image2_path']
                    if os.path.exists(img2_path):
                        self.image2 = cv2.imread(img2_path)
                        if self.image2 is not None:
                            self.image2_path = img2_path
                            self.display_image2 = self.image2.copy()
                            print(f"Auto-loaded Image 2: {os.path.basename(img2_path)}")
                
                # Update display
                self.update_display()
                
                print(f"Successfully loaded {len(self.point_pairs)} matching point pairs from {os.path.basename(file_path)}")
                
                # Show summary of loaded points
                if len(self.point_pairs) > 0:
                    print("Loaded point pairs:")
                    for i, (p1, p2) in enumerate(self.point_pairs):
                        print(f"  Pair {i+1}: Image1{p1} <-> Image2{p2}")
                
            except Exception as e:
                print(f"Failed to load points: {str(e)}")
                # Reset points on error
                self.points1 = []
                self.points2 = []
                self.point_pairs = []
                self.temp_point1 = None
                self.update_display()
    
    def export_points(self):
        """Export matching points to a file"""
        if len(self.point_pairs) == 0:
            print("No matching points to export!")
            return
        
        # Create a temporary tkinter root for file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        file_path = filedialog.asksaveasfilename(
            title="Export Matching Points",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("YAML files", "*.yaml")]
        )
        
        root.destroy()  # Clean up the temporary root
        
        if file_path:
            try:
                # Prepare data for export
                export_data = {
                    "image1_path": self.image1_path,
                    "image2_path": self.image2_path,
                    "image1_size": [int(self.image1.shape[1]), int(self.image1.shape[0])] if self.image1 is not None else None,
                    "image2_size": [int(self.image2.shape[1]), int(self.image2.shape[0])] if self.image2 is not None else None,
                    "matching_points": {
                        "points1": [[int(p[0]), int(p[1])] for p in self.points1],
                        "points2": [[int(p[0]), int(p[1])] for p in self.points2],
                        "point_pairs": [[[int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])]] for p1, p2 in self.point_pairs]
                    },
                    "total_pairs": len(self.point_pairs)
                }
                
                # Add homography matrix if computed
                if self.homography_matrix is not None:
                    export_data["homography"] = {
                        "matrix": self.homography_matrix.reshape(9).tolist(),
                        "computed": True
                    }
                else:
                    export_data["homography"] = {
                        "matrix": None,
                        "computed": False
                    }
                
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(export_data, f, indent=4)
                elif file_path.endswith('.yaml'):
                    import yaml
                    with open(file_path, 'w') as f:
                        yaml.dump(export_data, f, default_flow_style=False)
                else:
                    # Export as text file
                    with open(file_path, 'w') as f:
                        f.write(f"Image 1: {self.image1_path}\n")
                        f.write(f"Image 2: {self.image2_path}\n")
                        f.write(f"Total matching pairs: {len(self.point_pairs)}\n\n")
                        
                        for i, (p1, p2) in enumerate(self.point_pairs):
                            f.write(f"Pair {i+1}:\n")
                            f.write(f"  Image1: ({p1[0]}, {p1[1]})\n")
                            f.write(f"  Image2: ({p2[0]}, {p2[1]})\n\n")
                
                print(f"Matching points exported successfully!\nFile: {file_path}")
                print(f"Exported {len(self.point_pairs)} point pairs")
                
            except Exception as e:
                print(f"Failed to export points: {str(e)}")
    
    def run(self):
        """Run the application"""
        print("Dual Image Matching Application")
        print("Instructions:")
        print("1. Click 'Load Image 1' and 'Load Image 2' buttons to load images")
        print("2. Click 'Add Points' to enter point selection mode")
        print("3. Click corresponding points in both images to create pairs")
        print("4. Click 'Load Points' to load previously saved matching points")
        print("5. Click 'Delete Point' to remove a specific point pair by number")
        print("6. Click 'Compute Homography' to calculate transformation (needs 4+ points)")
        print("7. Click 'Export Points' to save the matching points")
        print("8. Press 'q' or close window to quit")
        print("Note: After computing homography, move mouse over Image 1 to see projected points")
        
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()


def main():
    app = DualImageMatchingApp()
    app.run()


if __name__ == "__main__":
    main()

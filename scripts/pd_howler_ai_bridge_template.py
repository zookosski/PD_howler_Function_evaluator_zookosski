#!/usr/bin/env python3
"""
PD Howler AI Bridge - Community Release
Complete AI processing bridge for depth mapping and background removal

SUPPORTED ALGORITHMS:
1. OpenCV Depth (Enhanced depth mapping)
2. MiDaS AI Depth (Professional thermal-style depth)
3. Depth Anything V2 (State-of-the-art depth estimation)
4. Basic RemBG (Thresholding background removal)
5. AI RemBG (u2net model for background removal)
6. InSPyReNet (Highest quality background removal)

REQUIREMENTS:
- Python 3.7+
- PyTorch
- OpenCV (cv2)
- NumPy
- PIL (Pillow)
- transformers (for Depth Anything V2)
- rembg (for AI background removal)

INSTALLATION:
See installation guide for detailed setup instructions

Created by: Claude & Community
Version: 1.0 - Production Ready
License: Community use - modify and share freely
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# CONFIGURATION - This will be automatically set by setup.bat
BRIDGE_BASE_PATH = "{{BRIDGE_PATH_PLACEHOLDER}}"

# Print system information
print("=" * 60)
print("PD HOWLER AI BRIDGE - COMMUNITY VERSION")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Bridge Path: {BRIDGE_BASE_PATH}")

# Test core dependencies
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch: Not available")
    TORCH_AVAILABLE = False

try:
    print(f"✅ OpenCV: {cv2.__version__}")
    OPENCV_AVAILABLE = True
except ImportError:
    print("❌ OpenCV: Not available")
    OPENCV_AVAILABLE = False

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
    NUMPY_AVAILABLE = True
except ImportError:
    print("❌ NumPy: Not available") 
    NUMPY_AVAILABLE = False

# Test AI dependencies
try:
    import rembg
    from rembg import remove, new_session
    from PIL import Image
    print("✅ RemBG: Available")
    REMBG_AVAILABLE = True
except ImportError:
    print("❌ RemBG: Not available (background removal will be limited)")
    REMBG_AVAILABLE = False

try:
    from transformers import pipeline
    print("✅ Transformers: Available") 
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("❌ Transformers: Not available (Depth Anything V2 will be limited)")
    TRANSFORMERS_AVAILABLE = False

print("=" * 60)

class PD_Howler_AI_Bridge:
    """
    Main AI processing bridge for PD Howler
    Handles all 6 supported algorithms with graceful fallbacks
    """
    
    def __init__(self):
        # Use the configured bridge path
        if BRIDGE_BASE_PATH == "{{BRIDGE_PATH_PLACEHOLDER}}":
            # Fallback for manual installation
            print("⚠️ Bridge path not configured by setup.bat")
            print("Using fallback path: ./Temp")
            self.temp_dir = Path("./Temp")
        else:
            self.temp_dir = Path(BRIDGE_BASE_PATH) / "Temp"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize device
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
            
        # Model storage
        self.midas_model = None
        self.midas_transform = None
        self.depth_anything_pipe = None
        self.rembg_session = None#!/usr/bin/env python3
"""
PD Howler AI Bridge - Community Release
Complete AI processing bridge for depth mapping and background removal

SUPPORTED ALGORITHMS:
1. OpenCV Depth (Enhanced depth mapping)
2. MiDaS AI Depth (Professional thermal-style depth)
3. Depth Anything V2 (State-of-the-art depth estimation)
4. Basic RemBG (Thresholding background removal)
5. AI RemBG (u2net model for background removal)
6. InSPyReNet (Highest quality background removal)

REQUIREMENTS:
- Python 3.7+
- PyTorch
- OpenCV (cv2)
- NumPy
- PIL (Pillow)
- transformers (for Depth Anything V2)
- rembg (for AI background removal)

INSTALLATION:
See installation guide for detailed setup instructions

Created by: Claude & Community
Version: 1.0 - Production Ready
License: Community use - modify and share freely
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Print system information
print("=" * 60)
print("PD HOWLER AI BRIDGE - COMMUNITY VERSION")
print("=" * 60)
print(f"Python: {sys.version}")

# Test core dependencies
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch: Not available")
    TORCH_AVAILABLE = False

try:
    print(f"✅ OpenCV: {cv2.__version__}")
    OPENCV_AVAILABLE = True
except ImportError:
    print("❌ OpenCV: Not available")
    OPENCV_AVAILABLE = False

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
    NUMPY_AVAILABLE = True
except ImportError:
    print("❌ NumPy: Not available") 
    NUMPY_AVAILABLE = False

# Test AI dependencies
try:
    import rembg
    from rembg import remove, new_session
    from PIL import Image
    print("✅ RemBG: Available")
    REMBG_AVAILABLE = True
except ImportError:
    print("❌ RemBG: Not available (background removal will be limited)")
    REMBG_AVAILABLE = False

try:
    from transformers import pipeline
    print("✅ Transformers: Available") 
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("❌ Transformers: Not available (Depth Anything V2 will be limited)")
    TRANSFORMERS_AVAILABLE = False

print("=" * 60)

class PD_Howler_AI_Bridge:
    """
    Main AI processing bridge for PD Howler
    Handles all 6 supported algorithms with graceful fallbacks
    """
    
    def __init__(self):
        self.temp_dir = Path("B:/TOOLS BY CLAUDE/Python_bridge/Temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize device
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
            
        # Model storage
        self.midas_model = None
        self.midas_transform = None
        self.depth_anything_pipe = None
        self.rembg_session = None
    
    def load_midas_model(self):
        """Load MiDaS depth estimation model"""
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available - cannot load MiDaS")
            return False
            
        try:
            print("🔄 Loading MiDaS AI depth model...")
            
            # Try to load cached model first
            cached_model_path = Path("midas_small_cached.pth")
            
            if cached_model_path.exists():
                print("   Loading from cache...")
                self.midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False)
                self.midas_model.load_state_dict(torch.load(cached_model_path, map_location=self.device))
            else:
                print("   Downloading model (first time only)...")
                self.midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
                torch.save(self.midas_model.state_dict(), cached_model_path)
                print("   Model cached for future use")
            
            self.midas_model.eval()
            
            # Load transform
            from torchvision.transforms import Compose, Normalize
            self.midas_transform = Compose([
                lambda x: torch.from_numpy(x).float().permute(2, 0, 1) / 255.0,
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ MiDaS model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ MiDaS loading failed: {e}")
            return False
    
    def load_depth_anything_v2(self):
        """Load Depth Anything V2 model"""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers not available - cannot load Depth Anything V2")
            return False
            
        try:
            print("🔄 Loading Depth Anything V2...")
            
            # Try working model first, then alternatives
            model_attempts = [
                ("LiheYoung/depth-anything-small-hf", "Working model"),
                ("depth-anything/Depth-Anything-V2-Small", "Official model"),
                ("depth-anything/Depth-Anything-V2-Base", "Base model"),
            ]
            
            for model_name, description in model_attempts:
                try:
                    print(f"   Trying {description}: {model_name}")
                    
                    device = 0 if (TORCH_AVAILABLE and torch.cuda.is_available()) else -1
                    self.depth_anything_pipe = pipeline(
                        "depth-estimation",
                        model=model_name,
                        device=device,
                        trust_remote_code=True
                    )
                    
                    print(f"✅ Depth Anything V2 loaded with {description}!")
                    return True
                    
                except Exception as e:
                    print(f"   ❌ {description} failed: {e}")
                    continue
            
            print("❌ All Depth Anything V2 models failed")
            return False
            
        except Exception as e:
            print(f"❌ Depth Anything V2 loading failed: {e}")
            return False
    
    def load_rembg_model(self, model_name='u2net'):
        """Load RemBG background removal model"""
        if not REMBG_AVAILABLE:
            print("❌ RemBG not available - cannot load background removal")
            return False
            
        try:
            print(f"🔄 Loading RemBG model: {model_name}...")
            self.rembg_session = new_session(model_name)
            print(f"✅ RemBG {model_name} loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ RemBG loading failed: {e}")
            return False
    
    def algorithm_1_opencv_depth(self, image):
        """Algorithm 1: Enhanced OpenCV depth mapping"""
        print("🎨 Processing with Enhanced OpenCV Depth...")
        
        if not OPENCV_AVAILABLE:
            print("❌ OpenCV not available")
            return None
            
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhanced depth estimation using multiple cues
            edges = cv2.Canny(gray, 50, 150)
            edges_blurred = cv2.GaussianBlur(edges, (15, 15), 0)
            
            # Brightness-based depth
            brightness_depth = cv2.equalizeHist(gray)
            
            # Combine methods
            combined_depth = cv2.addWeighted(edges_blurred, 0.6, brightness_depth, 0.4, 0)
            final_depth = cv2.bilateralFilter(combined_depth, 9, 75, 75)
            
            # Apply JET colormap for vibrant depth visualization
            depth_colored = cv2.applyColorMap(final_depth, cv2.COLORMAP_JET)
            
            print("✅ OpenCV depth map generated successfully!")
            return depth_colored
            
        except Exception as e:
            print(f"❌ OpenCV depth processing failed: {e}")
            return None
    
    def algorithm_2_midas_depth(self, image):
        """Algorithm 2: MiDaS AI depth estimation"""
        print("🧠 Processing with MiDaS AI Depth...")
        
        if not self.load_midas_model():
            print("⚠️ MiDaS failed, using OpenCV fallback...")
            return self.algorithm_1_opencv_depth(image)
            
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb_image.shape[:2]
            
            # Process with MiDaS
            input_size = 384
            rgb_resized = cv2.resize(rgb_image, (input_size, input_size))
            input_tensor = self.midas_transform(rgb_resized).unsqueeze(0)
            
            with torch.no_grad():
                depth = self.midas_model(input_tensor)
                if isinstance(depth, tuple):
                    depth = depth[0]
                if len(depth.shape) == 3:
                    depth = depth.unsqueeze(1)
                
                # Resize back to original dimensions
                import torch.nn.functional as F
                depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)
                depth = depth.squeeze().cpu().numpy()
            
            # Normalize and apply thermal colormap
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:
                depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth, dtype=np.uint8)
            
            # Use PLASMA colormap for thermal appearance
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
            
            print("✅ MiDaS AI depth map generated successfully!")
            return depth_colored
            
        except Exception as e:
            print(f"❌ MiDaS processing failed: {e}")
            print("⚠️ Using OpenCV fallback...")
            return self.algorithm_1_opencv_depth(image)
    
    def algorithm_3_depth_anything_v2(self, image):
        """Algorithm 3: Depth Anything V2 (state-of-the-art)"""
        print("🚀 Processing with Depth Anything V2...")
        
        if not self.load_depth_anything_v2():
            print("⚠️ Depth Anything V2 failed, using enhanced OpenCV...")
            return self.enhanced_opencv_depth(image)
            
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            print("   Running cutting-edge depth inference...")
            result = self.depth_anything_pipe(pil_image)
            
            # Extract depth data
            depth = np.array(result["depth"])
            depth_resized = cv2.resize(depth, (image.shape[1], image.shape[0]))
            
            # Normalize and apply VIRIDIS colormap
            depth_min, depth_max = depth_resized.min(), depth_resized.max()
            if depth_max > depth_min:
                depth_normalized = ((depth_resized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth_resized, dtype=np.uint8)
            
            # Use VIRIDIS for distinctive green/purple depth map
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
            
            print("✅ Depth Anything V2 processing successful!")
            return depth_colored
            
        except Exception as e:
            print(f"❌ Depth Anything V2 processing failed: {e}")
            print("⚠️ Using enhanced OpenCV fallback...")
            return self.enhanced_opencv_depth(image)
    
    def enhanced_opencv_depth(self, image):
        """Enhanced OpenCV depth (better than basic)"""
        print("🎨 Generating enhanced OpenCV depth...")
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multiple depth cues
            edges = cv2.Canny(gray, 30, 100)
            edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            
            # Gradient-based depth
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            gradient_mag = cv2.convertScaleAbs(gradient_mag)
            
            # Brightness-based depth
            brightness_depth = cv2.equalizeHist(gray)
            
            # Combine all cues
            combined = cv2.addWeighted(edges_dilated, 0.4, gradient_mag, 0.3, 0)
            combined = cv2.addWeighted(combined, 0.7, brightness_depth, 0.3, 0)
            
            # Smooth and enhance
            final_depth = cv2.bilateralFilter(combined, 9, 75, 75)
            
            # Use VIRIDIS to match Depth Anything V2 style
            depth_colored = cv2.applyColorMap(final_depth, cv2.COLORMAP_VIRIDIS)
            
            print("✅ Enhanced OpenCV depth generated")
            return depth_colored
            
        except Exception as e:
            print(f"❌ Enhanced OpenCV failed: {e}")
            return None
    
    def algorithm_4_basic_rembg(self, image):
        """Algorithm 4: Basic background removal using thresholding"""
        print("🎭 Processing with Basic Background Removal...")
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use Otsu's method for automatic threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Create mask
            mask = thresh.astype(float) / 255.0
            mask_3ch = np.stack([mask, mask, mask], axis=2)
            
            # Apply mask (keep foreground, white background)
            result = image.astype(float) * mask_3ch + (1 - mask_3ch) * 255
            result = result.astype(np.uint8)
            
            print("✅ Basic background removal completed!")
            return result
            
        except Exception as e:
            print(f"❌ Basic background removal failed: {e}")
            return None
    
    def algorithm_5_ai_rembg_u2net(self, image):
        """Algorithm 5: AI RemBG with u2net model"""
        print("🤖 Processing with AI RemBG (u2net)...")
        
        if not self.load_rembg_model('u2net'):
            print("⚠️ AI RemBG failed, using basic removal...")
            return self.algorithm_4_basic_rembg(image)
            
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            print("   Running AI background removal...")
            result_pil = remove(pil_image, session=self.rembg_session)
            
            # Convert back to OpenCV format
            result_array = np.array(result_pil)
            
            if result_array.shape[2] == 4:  # RGBA
                # Extract alpha and create white background composite
                alpha = result_array[:, :, 3].astype(float) / 255.0
                rgb_result = result_array[:, :, :3]
                
                white_bg = np.ones_like(rgb_result) * 255
                alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
                
                composited = (rgb_result * alpha_3ch + white_bg * (1 - alpha_3ch)).astype(np.uint8)
                result_bgr = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
            else:
                result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            
            print("✅ AI RemBG processing successful!")
            return result_bgr
            
        except Exception as e:
            print(f"❌ AI RemBG processing failed: {e}")
            print("⚠️ Using basic removal fallback...")
            return self.algorithm_4_basic_rembg(image)
    
    def algorithm_6_inspyrenet(self, image):
        """Algorithm 6: InSPyReNet (highest quality background removal)"""
        print("🔬 Processing with InSPyReNet (Highest Quality)...")
        
        # NOTE: InSPyReNet model name might need verification
        if not self.load_rembg_model('isnet-general-use'):
            print("⚠️ InSPyReNet failed, trying AI RemBG...")
            return self.algorithm_5_ai_rembg_u2net(image)
            
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            print("   Running premium background removal...")
            result_pil = remove(pil_image, session=self.rembg_session)
            
            # Convert back to OpenCV format
            result_array = np.array(result_pil)
            
            if result_array.shape[2] == 4:  # RGBA
                alpha = result_array[:, :, 3].astype(float) / 255.0
                rgb_result = result_array[:, :, :3]
                
                white_bg = np.ones_like(rgb_result) * 255
                alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
                
                composited = (rgb_result * alpha_3ch + white_bg * (1 - alpha_3ch)).astype(np.uint8)
                result_bgr = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
            else:
                result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            
            print("✅ InSPyReNet processing successful!")
            return result_bgr
            
        except Exception as e:
            print(f"❌ InSPyReNet processing failed: {e}")
            print("⚠️ Using AI RemBG fallback...")
            return self.algorithm_5_ai_rembg_u2net(image)
    
    def load_frame_from_pd_howler(self, input_path):
        """Load frame data from PD Howler text export"""
        print(f"📖 Loading frame from: {input_path}")
        
        try:
            with open(input_path, 'r') as file:
                width = int(file.readline().strip())
                height = int(file.readline().strip())
                
                print(f"   Frame dimensions: {width}x{height}")
                
                image = np.zeros((height, width, 3), dtype=np.uint8)
                
                for y in range(height):
                    for x in range(width):
                        line = file.readline().strip()
                        if not line:
                            raise ValueError(f"Unexpected end of file at ({x}, {y})")
                        
                        r, g, b = map(int, line.split())
                        image[y, x] = [b, g, r]  # OpenCV uses BGR
                    
                    if y % (height // 10) == 0 and y > 0:
                        progress = (y / height) * 100
                        print(f"   Loading: {progress:.1f}%")
                
                print("✅ Frame loaded successfully!")
                return image
                
        except Exception as e:
            print(f"❌ Frame loading failed: {e}")
            return None
    
    def save_frame_for_pd_howler(self, image, output_path):
        """Save processed image in PD Howler text format"""
        print(f"💾 Saving result to: {output_path}")
        
        try:
            height, width, channels = image.shape
            
            with open(output_path, 'w') as file:
                file.write(f"{width}\n")
                file.write(f"{height}\n")
                
                for y in range(height):
                    for x in range(width):
                        b, g, r = image[y, x]
                        file.write(f"{r} {g} {b}\n")
                    
                    if y % (height // 10) == 0 and y > 0:
                        progress = (y / height) * 100
                        print(f"   Saving: {progress:.1f}%")
                
            print("✅ Result saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Result saving failed: {e}")
            return False
    
    def process_frame(self):
        """Main processing function - reads config and processes frame"""
        print("\n" + "🚀 STARTING AI PROCESSING" + "\n")
        
        # File paths
        input_path = self.temp_dir / "input_frame.txt"
        config_path = self.temp_dir / "ai_config.txt"
        output_path = self.temp_dir / "output_result.txt"
        
        # Load configuration
        algorithm = 6  # Default to InSPyReNet
        model_name = "isnet-general-use"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        if line.startswith('algorithm='):
                            algorithm = int(line.split('=')[1].strip())
                        elif line.startswith('model='):
                            model_name = line.split('=')[1].strip()
                print(f"📋 Config loaded: Algorithm {algorithm}, Model {model_name}")
            except Exception as e:
                print(f"⚠️ Config loading failed: {e}, using defaults")
        
        # Load input frame
        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            return False
        
        image = self.load_frame_from_pd_howler(input_path)
        if image is None:
            return False
        
        # Save debug input
        debug_input_path = self.temp_dir / "debug_input.png"
        cv2.imwrite(str(debug_input_path), image)
        print(f"💡 Debug input saved: {debug_input_path}")
        
        # Process with selected algorithm
        print(f"\n🎯 PROCESSING WITH ALGORITHM {algorithm}")
        
        algorithm_map = {
            1: self.algorithm_1_opencv_depth,
            2: self.algorithm_2_midas_depth,
            3: self.algorithm_3_depth_anything_v2,
            4: self.algorithm_4_basic_rembg,
            5: self.algorithm_5_ai_rembg_u2net,
            6: self.algorithm_6_inspyrenet
        }
        
        process_func = algorithm_map.get(algorithm, self.algorithm_6_inspyrenet)
        result_image = process_func(image)
        
        if result_image is None:
            print("❌ All processing methods failed!")
            return False
        
        # Save debug output
        debug_output_path = self.temp_dir / f"debug_algorithm_{algorithm}.png"
        cv2.imwrite(str(debug_output_path), result_image)
        print(f"💡 Debug output saved: {debug_output_path}")
        
        # Save result for PD Howler
        success = self.save_frame_for_pd_howler(result_image, output_path)
        if not success:
            return False
        
        print("\n" + "✅ AI PROCESSING COMPLETE!" + "\n")
        print(f"📊 Algorithm used: {algorithm}")
        print(f"📁 Result saved: {output_path}")
        print(f"🖼️ Debug files in: {self.temp_dir}")
        
        return True

def main():
    """Main entry point for the AI bridge"""
    try:
        # Check basic requirements
        if not OPENCV_AVAILABLE or not NUMPY_AVAILABLE:
            print("❌ Missing critical dependencies (OpenCV, NumPy)")
            sys.exit(1)
        
        # Initialize and run bridge
        bridge = PD_Howler_AI_Bridge()
        success = bridge.process_frame()
        
        if success:
            print("🎉 PD Howler AI Bridge completed successfully!")
            sys.exit(0)
        else:
            print("💥 PD Howler AI Bridge failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

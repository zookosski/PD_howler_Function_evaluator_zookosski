# PD Howler AI Bridge - Installation Guide

**Complete AI depth mapping and background removal for PD Howler**

---

## 🎯 **What This Does**

Transform your PD Howler artwork with cutting-edge AI:

- **🎨 Depth Mapping**: Convert 2D drawings to 3D depth maps
- **🎭 Background Removal**: AI-powered background removal for clean subjects  
- **🚀 6 Algorithms**: From fast OpenCV to state-of-the-art AI models
- **🔄 Seamless Integration**: Works directly in PD Howler via Lua script

---

## 📋 **System Requirements**

### **Essential Requirements**
- **Windows 10/11** (recommended)
- **PD Howler** with Lua scripting support
- **Python 3.7+** (Python 3.9-3.11 recommended)
- **8GB+ RAM** (16GB+ for AI models)
- **5GB+ free disk space** (for models and dependencies)

### **Optional (for AI features)**
- **NVIDIA GPU** with CUDA support (faster processing)
- **High-speed internet** (for initial model downloads)

---

## 🛠️ **Installation Steps**

### **Step 1: Install Python Environment**

1. **Download Python 3.11** from [python.org](https://python.org)
2. **Important**: Check "Add Python to PATH" during installation
3. **Verify installation**:
   ```bash
   python --version
   pip --version
   ```

### **Step 2: Create Project Directory**

Create this exact folder structure:
```
B:\TOOLS BY CLAUDE\
├── Python_bridge\
│   ├── Python311\          # Portable Python (optional)
│   ├── Temp\              # Working files
│   ├── Output\             # Saved results
│   └── Models\             # Cached AI models
```

**Note**: You can change the base path from `B:\TOOLS BY CLAUDE\` to any location, but update all scripts accordingly.

### **Step 3: Install Python Dependencies**

Open Command Prompt and run:

```bash
# Core dependencies (required)
pip install opencv-python numpy pillow

# AI dependencies (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU support (if you have NVIDIA GPU):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Advanced AI models
pip install transformers
pip install rembg
```

### **Step 4: Install PD Howler Bridge Files**

1. **Copy Python script**: Save `pd_howler_ai_bridge.py` to `B:\TOOLS BY CLAUDE\Python_bridge\`
2. **Copy Lua script**: Save `1_Depth_REMBG_PD.lua` to your PD Howler Scripts folder
   - Usually: `C:\Program Files\Project Dogwaffle\Scripts\`

### **Step 5: Test Installation**

1. **Open PD Howler**
2. **Create or open an image**
3. **Run**: Filter > Scripts > 1_Depth_REMBG_PD
4. **Select algorithm 1** (OpenCV - fastest)
5. **Click OK** and wait for processing

---

## 🎮 **Usage Guide**

### **Algorithm Selection Guide**

| Algorithm | Type | Speed | Quality | Use Case |
|-----------|------|-------|---------|----------|
| **1 - OpenCV** | Depth mapping | ⚡ Very Fast | 🟡 Good | Quick depth previews |
| **2 - MiDaS** | AI depth | 🐌 Slow | 🟢 High | Professional depth maps |
| **3 - Depth Anything V2** | AI depth | 🐌 Slow | 🔵 Highest | State-of-the-art depth |
| **4 - Basic RemBG** | Background removal | ⚡ Fast | 🟡 Basic | Simple backgrounds |
| **5 - AI RemBG** | AI background removal | 🐌 Slow | 🟢 High | Complex subjects |
| **6 - InSPyReNet** | Premium background removal | 🐌 Slow | 🔵 Highest | Professional results |

### **Workflow Recommendations**

#### **For Blender Integration** 📦
1. Use **Algorithm 2 or 3** for depth maps
2. Import depth map as displacement in Blender
3. Create 3D geometry from your 2D art

#### **For Character Art** 🎭
1. Use **Algorithm 5 or 6** for background removal
2. Get clean character on transparent/white background
3. Perfect for compositing and animation

#### **For Quick Tests** ⚡
1. Use **Algorithm 1** (OpenCV) for rapid iteration
2. Switch to AI algorithms for final output

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **"Cannot connect to PD Howler"**
- Make sure PD Howler is running
- Check that Lua scripting is enabled
- Verify script is in correct Scripts folder

#### **"Python processing failed"**
- Check Python installation: `python --version`
- Verify dependencies: `pip list`
- Check file paths in both Lua and Python scripts

#### **"AI models not loading"**
- First run downloads models (may take 5-10 minutes)
- Check internet connection
- Ensure sufficient disk space (5GB+)

#### **"Permission denied" errors**
- Run PD Howler as Administrator (once)
- Check folder permissions for bridge directory
- Verify antivirus isn't blocking Python execution

### **Performance Issues**

#### **Processing too slow**
- Use Algorithm 1 (OpenCV) for testing
- Consider GPU installation for AI algorithms
- Close other applications to free RAM

#### **Out of memory errors**
- Reduce image size in PD Howler before processing
- Use CPU instead of GPU for large images
- Close unnecessary applications

---

## 📁 **File Structure Reference**

```
B:\TOOLS BY CLAUDE\Python_bridge\
├── pd_howler_ai_bridge.py      # Main Python processor
├── requirements.txt            # Dependencies list
├── Temp\
│   ├── input_frame.txt        # From PD Howler
│   ├── ai_config.txt          # Processing settings
│   ├── output_result.txt      # To PD Howler
│   ├── debug_input.png        # Visual debugging
│   ├── debug_algorithm_*.png  # Result previews
│   └── status.txt             # Processing status
├── Output\                    # Saved PNG files
└── Models\                    # Cached AI models
    ├── midas_small_cached.pth
    └── [various RemBG models]
```

---

## 🔄 **Advanced Configuration**

### **Custom Paths**

If you need to use different paths, update these files:

**In Lua script** (`1_Depth_REMBG_PD.lua`):
```lua
-- Change this line:
local export_path = "B:\\TOOLS BY CLAUDE\\Python_bridge\\Temp\\input_frame.txt"
-- To your path:
local export_path = "C:\\YourPath\\Python_bridge\\Temp\\input_frame.txt"
```

**In Python script** (`pd_howler_ai_bridge.py`):
```python
# Change this line:
self.temp_dir = Path("B:/TOOLS BY CLAUDE/Python_bridge/Temp")
# To your path:
self.temp_dir = Path("C:/YourPath/Python_bridge/Temp")
```

### **GPU Configuration**

For NVIDIA GPU support:
```bash
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU detection:
```python
import torch
print(torch.cuda.is_available())  # Should print True
```

---

## 🎨 **Integration Examples**

### **Blender Workflow**
1. Create artwork in PD Howler
2. Run Algorithm 2 or 3 for depth map
3. In Blender:
   - Import depth map as image texture
   - Use as displacement modifier
   - Apply to subdivided plane
   - Render 3D scene

### **ComfyUI Workflow**
1. Create base artwork in PD Howler
2. Run Algorithm 5 or 6 for clean subject
3. In ComfyUI:
   - Load cleaned subject as input
   - Apply style transfer or enhancement
   - Generate variations or backgrounds

### **Reaper Audio Sync**
1. Process character art with background removal
2. Sync with audio timeline in Reaper
3. Use Python scripts to coordinate timing
4. Export synchronized media for video projects

---

## 🤝 **Community & Support**

### **Sharing Your Results**
- Post examples and improvements to GitHub
- Share workflow tips and custom configurations
- Help others troubleshoot installation issues

### **Contributing**
- Report bugs and suggest features
- Improve documentation and guides
- Create additional algorithm integrations

### **Getting Help**
- Check troubleshooting section first
- Verify all installation steps completed
- Test with simple images before complex artwork
- Share error messages for specific debugging

---

## 📜 **License & Credits**

**License**: Community use - modify and share freely

**Credits**:
- **Created by**: Claude & Community
- **PD Howler**: Project Dogwaffle by Dan Ritchie
- **AI Models**: 
  - MiDaS by Intel ISL
  - Depth Anything V2 by LiheYoung
  - RemBG by danielgatis
  - InSPyReNet by GewelsJI

**Special Thanks**: PD Howler community for testing and feedback

---

## 🚀 **What's Next**

This bridge opens up endless creative possibilities:

- **3D Art Pipeline**: PD Howler → Depth AI → Blender → Render
- **Character Animation**: PD Howler → Background Removal → Animation Software
- **AI Enhancement**: PD Howler → Clean Subject → AI Style Transfer
- **Multi-Media Projects**: Coordinate with audio in Reaper for complete productions

**Happy Creating!** 🎨✨

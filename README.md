# PD_howler_Function_evaluator_zookosski
A collection of Function evaluator scripts for PD howler 

PD Howler AI Bridge 🎨🤖
Transform your PD Howler artwork with cutting-edge AI depth mapping and background removal

🚀 What This Does
Bridge the gap between traditional 2D art and modern AI processing by bringing 6 powerful algorithms directly into PD Howler:

🎨 Depth Mapping: Convert drawings to 3D depth maps for Blender integration
🎭 Background Removal: AI-powered subject extraction for clean compositing
⚡ Multiple Options: From fast OpenCV to state-of-the-art AI models
🔄 Seamless Workflow: Works directly in PD Howler via simple Lua script

Show Image
Example: Original artwork → AI depth map → 3D Blender scene

🎯 Quick Start
1. Download & Install
bashgit clone https://github.com/your-username/pd-howler-ai-bridge.git
cd pd-howler-ai-bridge
setup.bat  # Windows automated installation
2. Copy Files

Copy pd_howler_ai_bridge.py to B:\TOOLS BY CLAUDE\Python_bridge\
Copy 1_Depth_REMBG_PD.lua to your PD Howler Scripts folder

3. Use in PD Howler

Open/create artwork in PD Howler
Run: Filter > Scripts > 1_Depth_REMBG_PD
Choose algorithm and enjoy AI processing!


🧠 Supported Algorithms
AlgorithmTypeQualitySpeedBest ForOpenCV DepthEnhanced depth mappingGood⚡ FastQuick previewsMiDaS AINeural depth estimationHigh🐌 SlowProfessional depthDepth Anything V2State-of-the-art depthHighest🐌 SlowBest quality depthBasic RemBGThreshold background removalBasic⚡ FastSimple backgroundsAI RemBG (u2net)Neural background removalHigh🐌 SlowComplex subjectsInSPyReNetPremium background removalHighest🐌 SlowProfessional results

🛠️ Installation
Requirements

PD Howler with Lua scripting
Python 3.7+ (3.9-3.11 recommended)
8GB+ RAM (16GB+ for AI models)
Windows 10/11 (primary support)

Automated Installation
bash# Download repository
git clone https://github.com/your-username/pd-howler-ai-bridge.git

# Run automated setup
cd pd-howler-ai-bridge
setup.bat
Manual Installation
bash# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
📖 Full installation guide: INSTALLATION.md

🎨 Creative Workflows
🎬 3D Animation Pipeline
PD Howler Art → AI Depth Map → Blender 3D → Animation

Create 2D artwork in PD Howler
Generate depth map with MiDaS or Depth Anything V2
Import depth as displacement in Blender
Animate 3D scene with camera movements

🎭 Character Isolation
PD Howler Character → AI Background Removal → Clean Subject

Draw character in PD Howler
Use AI RemBG or InSPyReNet for clean extraction
Perfect for compositing, animation, or AI style transfer

🔄 Iterative Enhancement
Sketch → AI Process → Refine → AI Process → Final

Start with rough sketch in PD Howler
Use OpenCV for quick depth feedback
Refine artwork based on depth visualization
Final pass with premium AI algorithms


📁 Project Structure
pd-howler-ai-bridge/
├── README.md                    # This file
├── INSTALLATION.md              # Detailed setup guide
├── requirements.txt             # Python dependencies
├── setup.bat                    # Automated Windows setup
├── pd_howler_ai_bridge.py      # Main Python processor
├── 1_Depth_REMBG_PD.lua        # PD Howler Lua script
├── examples/                    # Example images and results
├── docs/                        # Additional documentation
└── LICENSE                      # Community license

🔧 Configuration
Custom Paths
Edit paths in both scripts to match your setup:
Lua script:
lualocal export_path = "C:\\YourPath\\Python_bridge\\Temp\\input_frame.txt"
Python script:
pythonself.temp_dir = Path("C:/YourPath/Python_bridge/Temp")
Algorithm Selection
Default algorithm can be changed in the Lua script:
lualocal algorithm = 6  -- 1=OpenCV, 6=InSPyReNet

🐛 Troubleshooting
Common Issues
"Cannot connect to PD Howler"

Ensure PD Howler is running
Check Lua script is in Scripts folder
Verify Lua scripting is enabled

"Python processing failed"

Test Python: python --version
Check dependencies: pip list
Verify file paths match your installation

AI models not loading

First run downloads models (5-10 minutes)
Ensure stable internet connection
Check available disk space (5GB+)

📖 Full troubleshooting guide: INSTALLATION.md#troubleshooting

🤝 Contributing
We welcome contributions from the community!
Ways to Contribute

🐛 Report bugs and suggest features
📚 Improve documentation and guides
🎨 Share example workflows and results
🔧 Add new algorithm integrations
💡 Optimize performance and compatibility

Development Setup
bashgit clone https://github.com/your-username/pd-howler-ai-bridge.git
cd pd-howler-ai-bridge
pip install -r requirements.txt
# Make your changes and submit PR

📜 Credits & License
License
Community use - modify and share freely. See LICENSE for details.
Credits

Created by: Claude & Community
PD Howler: Project Dogwaffle by Dan Ritchie
AI Models:

MiDaS by Intel ISL
Depth Anything V2 by LiheYoung
RemBG by danielgatis
InSPyReNet by GewelsJI



Special Thanks

PD Howler community for testing and feedback
AI researchers for open-source model contributions
Bridge workflow pioneers in creative coding


🌟 Showcase
Share your creations using #PDHowlerAI and we'll feature them here!

📞 Support & Community

🐛 Issues: GitHub Issues
💬 Discussions: GitHub Discussions
📧 Email: support@pd-howler-ai-bridge.com
🎨 Gallery: Community Showcase


Transform your art with AI - Bridge the future of creativity! 🎨✨

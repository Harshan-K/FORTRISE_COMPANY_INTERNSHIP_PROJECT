#!/usr/bin/env python3
"""
Setup script for AI-Powered Question Paper Generator
This script helps initialize the project and install dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🎓 AI-Powered Question Paper Generator Setup             ║
║                                                              ║
║    Setting up RAG + FAISS + Llama + LangChain + CrewAI     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("📋 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")

def create_virtual_environment():
    """Create virtual environment"""
    print("\n🔧 Setting up virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("⚠️  Virtual environment already exists. Skipping creation.")
        return
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        sys.exit(1)

def get_activation_command():
    """Get the correct activation command based on OS"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Determine python executable in venv
    if platform.system() == "Windows":
        python_exe = "venv\\Scripts\\python.exe"
        pip_exe = "venv\\Scripts\\pip.exe"
    else:
        python_exe = "venv/bin/python"
        pip_exe = "venv/bin/pip"
    
    try:
        # Upgrade pip first
        print("   Upgrading pip...")
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements
        print("   Installing project dependencies...")
        subprocess.run([pip_exe, "install", "-r", "requirements.txt"], 
                      check=True)
        
        print("✅ Dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("\n💡 Try manual installation:")
        print(f"   1. Activate environment: {get_activation_command()}")
        print("   2. Install dependencies: pip install -r requirements.txt")
        sys.exit(1)

def create_directories():
    """Create necessary project directories"""
    print("\n📁 Creating project directories...")
    
    try:
        # Import config to create directories
        from config import Config
        Config.create_directories()
        print("✅ Project directories created")
        
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        
        # Fallback: create manually
        directories = ["uploads", "vectorstore", "database", "exports"]
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
        print("✅ Directories created manually")

def initialize_database():
    """Initialize SQLite database"""
    print("\n💾 Initializing database...")
    
    try:
        from database.db_manager import DatabaseManager
        db_manager = DatabaseManager()
        print("✅ Database initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")

def verify_installation():
    """Verify that all components are working"""
    print("\n🔍 Verifying installation...")
    
    try:
        # Test imports
        print("   Testing core imports...")
        from config import Config
        from database.db_manager import DatabaseManager
        from utils.document_processor import DocumentProcessor
        from vectorstore.faiss_store import VectorStore
        
        print("✅ All core components imported successfully")
        
        # Test vector store
        print("   Testing vector store...")
        vector_store = VectorStore()
        stats = vector_store.get_stats()
        print(f"   Vector store initialized: {stats['total_documents']} documents")
        
        print("✅ Installation verification completed")
        
    except Exception as e:
        print(f"⚠️  Verification warning: {e}")
        print("   The application should still work, but some features might be limited")

def print_next_steps():
    """Print instructions for running the application"""
    activation_cmd = get_activation_command()
    
    next_steps = f"""
🎉 Setup completed successfully!

📋 Next Steps:
   1. Activate virtual environment:
      {activation_cmd}
   
   2. Run the application:
      python app.py
   
   3. Open your browser and go to:
      http://localhost:7860
   
   4. Upload academic documents and start generating question papers!

📚 Documentation:
   - Check README.md for detailed usage instructions
   - View config.py for configuration options
   
🆘 Need help?
   - Check the troubleshooting section in README.md
   - Ensure all dependencies are properly installed
   
💡 Tips:
   - Start by uploading a few PDF documents
   - Try different difficulty levels and Bloom's taxonomy levels
   - Use the analytics dashboard to monitor performance
"""
    
    print(next_steps)

def main():
    """Main setup function"""
    print_banner()
    
    # Verify we're in the correct directory
    if not Path("requirements.txt").exists():
        print("❌ Error: requirements.txt not found!")
        print("   Please run this script from the project root directory")
        sys.exit(1)
    
    try:
        # Setup steps
        check_python_version()
        create_virtual_environment()
        install_dependencies()
        create_directories()
        initialize_database()
        verify_installation()
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        print("\n💡 Try running individual setup steps manually:")
        print("   1. Create virtual environment: python -m venv venv")
        print(f"   2. Activate environment: {get_activation_command()}")
        print("   3. Install dependencies: pip install -r requirements.txt")
        print("   4. Run application: python app.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
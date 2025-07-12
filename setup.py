#!/usr/bin/env python3
"""
YouTube Analytics Setup Script
Automates the installation and setup of the YouTube Analytics project.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python 3.7+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install required Python packages"""
    print("\n📦 INSTALLING DEPENDENCIES")
    print("="*50)
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print("✅ pip is available")
    except subprocess.CalledProcessError:
        print("❌ pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        cmd = f"{sys.executable} -m pip install -r requirements.txt"
        return run_command(cmd, "Installing dependencies from requirements.txt")
    else:
        print("❌ requirements.txt not found. Installing basic dependencies...")
        basic_packages = [
            "pandas", "plotly", "streamlit", "dash", 
            "scikit-learn", "openpyxl", "jupyter"
        ]
        
        for package in basic_packages:
            cmd = f"{sys.executable} -m pip install {package}"
            run_command(cmd, f"Installing {package}")
        
        return True

def verify_installation():
    """Verify that all required packages are installed"""
    print("\n🔍 VERIFYING INSTALLATION")
    print("="*50)
    
    required_packages = [
        "pandas", "plotly", "streamlit", "dash", 
        "sklearn", "openpyxl"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ {package} import failed")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages manually:")
        for package in failed_imports:
            print(f"   pip install {package}")
        return False
    
    print("\n✅ All packages verified successfully!")
    return True

def check_data_files():
    """Check if sample data files exist"""
    print("\n📊 CHECKING DATA FILES")
    print("="*50)
    
    data_files = ["videos.csv", "subscribers.csv"]
    missing_files = []
    
    for file in data_files:
        if Path(file).exists():
            print(f"✅ {file} found")
        else:
            print(f"⚠️  {file} not found")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n📝 Sample data files will be created during first run.")
        print("You can replace them with your actual YouTube Studio export data.")
    
    return True

def display_usage_instructions():
    """Display instructions for using the project"""
    print("\n🚀 USAGE INSTRUCTIONS")
    print("="*60)
    print("1. Basic Analytics:")
    print("   python youtube_analytics.py")
    print()
    print("2. Interactive Streamlit Dashboard:")
    print("   streamlit run streamlit_dashboard.py")
    print("   Then open: http://localhost:8501")
    print()
    print("3. Dash Dashboard:")
    print("   python dash_dashboard.py")
    print("   Then open: http://localhost:8050")
    print()
    print("4. Jupyter Notebook (coming soon):")
    print("   jupyter notebook")
    print()
    print("📁 DATA FILES:")
    print("- Replace videos.csv with your YouTube Studio video data")
    print("- Replace subscribers.csv with your YouTube Studio subscriber data")
    print()
    print("📋 YOUTUBE STUDIO DATA EXPORT:")
    print("1. Go to YouTube Studio → Analytics")
    print("2. Click on 'Export Report' tab")
    print("3. Select date range and metrics")
    print("4. Export as CSV and save in this directory")
    print()
    print("🎯 FEATURES:")
    print("- Interactive visualizations with Plotly")
    print("- Subscriber growth analysis")
    print("- Engagement rate calculations")
    print("- Machine learning predictions")
    print("- Export reports to Excel")
    print("- Real-time dashboard updates")

def main():
    """Main setup function"""
    print("🎬 YOUTUBE ANALYTICS PROJECT SETUP")
    print("="*60)
    print("Setting up your YouTube Analytics dashboard...")
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    print()
    if not install_requirements():
        print("❌ Failed to install dependencies. Please check the errors above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed.")
        sys.exit(1)
    
    # Check data files
    check_data_files()
    
    # Display usage instructions
    display_usage_instructions()
    
    print("\n🎉 SETUP COMPLETE!")
    print("="*60)
    print("Your YouTube Analytics project is ready to use!")
    print("Start with: python youtube_analytics.py")

if __name__ == "__main__":
    main()

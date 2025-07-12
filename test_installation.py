#!/usr/bin/env python3
"""
Test script to verify YouTube Analytics installation
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            module = importlib.import_module(module_name)
            return True, f"✅ {package_name} installed correctly"
        else:
            importlib.import_module(module_name)
            return True, f"✅ {module_name} imported successfully"
    except ImportError as e:
        package = package_name or module_name
        return False, f"❌ {package} not found: {e}"

def main():
    """Test all required dependencies"""
    print("🧪 TESTING YOUTUBE ANALYTICS INSTALLATION")
    print("="*50)
    
    # Test Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 7:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")
    else:
        print(f"❌ Python 3.7+ required. Current: {python_version.major}.{python_version.minor}")
        return False
    
    # Test required packages
    packages = [
        ("pandas", "Pandas"),
        ("plotly", "Plotly"),
        ("plotly.express", "Plotly Express"),
        ("plotly.graph_objects", "Plotly Graph Objects"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("streamlit", "Streamlit"),
        ("dash", "Dash"),
        ("openpyxl", "OpenPyXL"),
    ]
    
    all_passed = True
    
    print("\n📦 Testing package imports:")
    for module, name in packages:
        success, message = test_import(module, name)
        print(f"  {message}")
        if not success:
            all_passed = False
    
    # Test project files
    print("\n📁 Testing project files:")
    from pathlib import Path
    
    required_files = [
        "youtube_analytics.py",
        "streamlit_dashboard.py", 
        "dash_dashboard.py",
        "requirements.txt",
        "videos.csv",
        "subscribers.csv"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file} found")
        else:
            print(f"  ❌ {file} missing")
            all_passed = False
    
    # Test main analytics import
    print("\n🔧 Testing main analytics module:")
    try:
        from youtube_analytics import YouTubeAnalytics
        analytics = YouTubeAnalytics()
        print("  ✅ YouTubeAnalytics class imported successfully")
        
        # Test data loading
        analytics.load_data()
        print("  ✅ Sample data loaded successfully")
        print(f"    📊 Videos: {len(analytics.videos_df)}")
        if analytics.subscribers_df is not None:
            print(f"    👥 Subscribers: {len(analytics.subscribers_df)}")
        
    except Exception as e:
        print(f"  ❌ Main module test failed: {e}")
        all_passed = False
    
    # Final result
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ YouTube Analytics is ready to use!")
        print("\n🚀 Next steps:")
        print("  • Run: python demo.py")
        print("  • Or: python youtube_analytics.py")
        print("  • Or: streamlit run streamlit_dashboard.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please fix the issues above before proceeding.")
        print("\n💡 Try running: python setup.py")
    
    print("="*50)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

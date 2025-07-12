#!/usr/bin/env python3
"""
YouTube Analytics Demo Script
Demonstrates all features of the YouTube Analytics project with sample data.
"""

import os
import sys
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n📍 Step {step}: {description}")
    print("-" * 40)

def main():
    """Main demo function"""
    print_header("🎬 YOUTUBE ANALYTICS PROJECT DEMO")
    print("Welcome to the YouTube Analytics demonstration!")
    print("This demo will showcase all features using sample data.")
    
    # Check if we're in the right directory
    if not Path("youtube_analytics.py").exists():
        print("❌ Please run this demo from the project root directory.")
        sys.exit(1)
    
    print_step(1, "Loading Analytics Engine")
    try:
        from youtube_analytics import YouTubeAnalytics
        analytics = YouTubeAnalytics()
        analytics.load_data()
        print("✅ Analytics engine loaded successfully!")
        print(f"📊 Found {len(analytics.videos_df)} videos in dataset")
        if analytics.subscribers_df is not None:
            print(f"👥 Found {len(analytics.subscribers_df)} subscriber records")
    except Exception as e:
        print(f"❌ Error loading analytics: {e}")
        return
    
    print_step(2, "Displaying Summary Statistics")
    analytics.display_summary_stats()
    
    print_step(3, "Creating Interactive Visualizations")
    print("🔄 Generating charts... (check your browser)")
    
    # Create all visualizations
    charts = [
        ("Views Over Time", analytics.create_views_over_time_chart),
        ("Engagement Comparison", analytics.create_engagement_comparison_chart),
        ("Engagement Rates", analytics.create_engagement_rates_chart),
        ("Performance Heatmap", analytics.create_performance_heatmap),
    ]
    
    for name, chart_func in charts:
        try:
            fig = chart_func()
            if fig:
                print(f"  ✅ {name} chart created")
                fig.show()
                time.sleep(1)  # Brief pause between charts
            else:
                print(f"  ⚠️  {name} chart skipped (no data)")
        except Exception as e:
            print(f"  ❌ {name} chart failed: {e}")
    
    # Subscriber chart if available
    if analytics.subscribers_df is not None:
        try:
            fig = analytics.create_subscriber_activity_chart()
            if fig:
                print("  ✅ Subscriber Activity chart created")
                fig.show()
        except Exception as e:
            print(f"  ❌ Subscriber Activity chart failed: {e}")
    
    print_step(4, "Running Machine Learning Predictions")
    try:
        model = analytics.predict_views()
        if model:
            print("✅ ML model trained successfully!")
        else:
            print("⚠️  ML model training skipped (insufficient data)")
    except Exception as e:
        print(f"❌ ML prediction failed: {e}")
    
    print_step(5, "Exporting Analysis Report")
    try:
        analytics.export_analysis_report('demo_report.xlsx')
        print("✅ Excel report exported as 'demo_report.xlsx'")
    except Exception as e:
        print(f"❌ Report export failed: {e}")
    
    print_step(6, "Dashboard Options")
    print("🌐 Available dashboard options:")
    print("\n1. Streamlit Dashboard:")
    print("   streamlit run streamlit_dashboard.py")
    print("   → http://localhost:8501")
    
    print("\n2. Dash Dashboard:")
    print("   python dash_dashboard.py")
    print("   → http://localhost:8050")
    
    print("\n3. Jupyter Notebook:")
    print("   jupyter notebook youtube_analytics_notebook.ipynb")
    
    # Ask user if they want to launch a dashboard
    print("\n" + "="*60)
    print("🚀 LAUNCH DASHBOARD?")
    print("="*60)
    
    choice = input("\nWould you like to launch a dashboard? (s)treamlit/(d)ash/(n)o: ").lower().strip()
    
    if choice == 's':
        print("\n🚀 Launching Streamlit dashboard...")
        print("📊 Opening http://localhost:8501 in your browser...")
        os.system("streamlit run streamlit_dashboard.py")
    elif choice == 'd':
        print("\n🚀 Launching Dash dashboard...")
        print("📊 Opening http://localhost:8050 in your browser...")
        try:
            from dash_dashboard import run_dashboard
            run_dashboard(debug=False)
        except ImportError:
            print("❌ Dash not installed. Install with: pip install dash")
        except Exception as e:
            print(f"❌ Dashboard launch failed: {e}")
    else:
        print("\n✅ Demo completed!")
    
    print_header("🎉 DEMO SUMMARY")
    print("✅ All core features demonstrated successfully!")
    print("\n📋 What you've seen:")
    print("  • Data loading and preprocessing")
    print("  • Summary statistics and insights")
    print("  • Interactive Plotly visualizations")
    print("  • Machine learning predictions")
    print("  • Automated report export")
    print("  • Dashboard options")
    
    print("\n🔧 Next steps:")
    print("  1. Replace sample CSV files with your YouTube Studio data")
    print("  2. Customize config.ini for your preferences")
    print("  3. Run regular analysis to track channel growth")
    print("  4. Use predictions to optimize content strategy")
    
    print("\n📚 For detailed usage, see README.md")
    print("🚀 Happy analyzing your YouTube data! 📺✨")

if __name__ == "__main__":
    main()

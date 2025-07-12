# 📺 YouTube Studio Data Analytics

A comprehensive Python application for analyzing YouTube Studio data with interactive visualizations, machine learning predictions, and automated reporting.

## 🚀 Features

- **📊 Interactive Dashboards**: Web-based dashboards using Streamlit and Dash
- **📈 Time Series Analysis**: Views, likes, and comments trends over time
- **🎯 Engagement Analytics**: Like rates, comment rates, and engagement patterns
- **👥 Subscriber Insights**: Growth and decline analysis with detailed metrics
- **🤖 ML Predictions**: Machine learning models to predict video performance
- **📁 Export Reports**: Automated Excel and PDF report generation
- **🔥 Correlation Analysis**: Performance correlation heatmaps
- **📱 Responsive Design**: Works on desktop and mobile devices

## 📋 Project Structure

```
YouTubeStudioDataAnalytics/
├── 📄 youtube_analytics.py          # Main analytics engine
├── 🌐 streamlit_dashboard.py        # Streamlit web dashboard
├── 🌐 dash_dashboard.py            # Dash web dashboard  
├── 📓 youtube_analytics_notebook.ipynb # Jupyter notebook for analysis
├── ⚙️ setup.py                     # Automated setup script
├── 📊 videos.csv                   # Sample video data
├── 👥 subscribers.csv              # Sample subscriber data
├── 📦 requirements.txt             # Python dependencies
├── ⚙️ config.ini                   # Configuration settings
└── 📖 README.md                    # This file
```

## 🛠️ Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/DuongCaoNhan/YouTubeStudioDataAnalytics.git
cd YouTubeStudioDataAnalytics

# Run automated setup
python setup.py
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install pandas plotly streamlit dash scikit-learn openpyxl jupyter
```

## 📊 Getting Your YouTube Data

### From YouTube Studio:

1. **Go to YouTube Studio** → Analytics
2. **Click "Export Report"** tab
3. **Select date range** and metrics
4. **Export as CSV** files:
   - `videos.csv`: Video performance data
   - `subscribers.csv`: Subscriber activity data

### Required CSV Columns:

**videos.csv:**
```
Title, Publish Date, Views, Likes, Comments, Duration (minutes)
```

**subscribers.csv:**
```
Date, Subscribers Gained, Subscribers Lost, Net Subscribers
```

## 🚀 Usage Options

### 1. 📊 Basic Analytics Script

```bash
python youtube_analytics.py
```

**Features:**
- Complete analytics pipeline
- Interactive Plotly charts
- Summary statistics
- ML predictions
- Excel report export

### 2. 🌐 Streamlit Web Dashboard

```bash
streamlit run streamlit_dashboard.py
```

**Then open:** http://localhost:8501

**Features:**
- Real-time interactive dashboard
- Multi-page navigation
- Date range filtering
- Live data updates
- Mobile-responsive design

### 3. 🌐 Dash Web Dashboard

```bash
python dash_dashboard.py
```

**Then open:** http://localhost:8050

**Features:**
- Professional dashboard interface
- Advanced interactivity
- Custom styling
- Real-time updates

### 4. 📓 Jupyter Notebook Analysis

```bash
jupyter notebook youtube_analytics_notebook.ipynb
```

**Features:**
- Step-by-step analysis
- Detailed explanations
- Interactive exploration
- Customizable analysis

## 📈 Key Analytics Features

### 📊 Video Performance Metrics

- **Views Analysis**: Track view patterns over time
- **Engagement Rates**: Like rate and comment rate calculations  
- **Performance Comparison**: Compare videos side-by-side
- **Top Performers**: Identify your best-performing content

### 👥 Subscriber Analytics

- **Growth Tracking**: Monitor subscriber gains and losses
- **Activity Patterns**: Analyze subscriber behavior over time
- **Net Growth**: Calculate overall subscriber growth trends

### 🤖 Machine Learning Predictions

- **View Prediction**: Predict views for new videos
- **Feature Importance**: Understand what drives performance
- **Model Accuracy**: R² scores and performance metrics
- **Interactive Predictor**: Test different video parameters

### 📊 Visualization Types

- **Line Charts**: Time series trends
- **Bar Charts**: Performance comparisons
- **Scatter Plots**: Correlation analysis
- **Heatmaps**: Performance correlation matrices
- **Histograms**: Distribution analysis
- **Bubble Charts**: Multi-dimensional analysis

## 🎯 Sample Insights You'll Get

### 📈 Performance Insights
- Which video topics perform best
- Optimal video length for your audience
- Best publishing times and patterns
- Engagement rate benchmarks

### 🎬 Content Strategy
- Content types that drive most engagement
- Correlation between video length and performance
- Subscriber impact of different content types
- Seasonal performance patterns

### 📊 Predictive Analytics
- Expected performance for new videos
- Feature importance for video success
- Performance confidence intervals
- Optimization recommendations

## ⚙️ Configuration

Edit `config.ini` to customize:

```ini
[data_files]
videos_file = your_videos.csv
subscribers_file = your_subscribers.csv

[dashboard]
streamlit_port = 8501
dash_port = 8050

[visualization]
youtube_red = #FF0000
chart_height = 500
```

## 📁 Export Options

### Excel Reports
- Comprehensive analytics data
- Summary statistics
- Top performers lists
- Subscriber activity data

### PDF Reports (Coming Soon)
- Executive summary
- Key visualizations
- Performance insights
- Recommendations

## 🔧 Advanced Features

### 🎮 Interactive Prediction Tool
Test how different parameters affect predicted views:
- Video duration
- Expected engagement rates
- Content type factors

### 📊 Custom Metrics
- Engagement Rate = (Likes + Comments) / Views × 100
- Like Rate = Likes / Views × 100  
- Comment Rate = Comments / Views × 100
- Performance Score = Combined weighted metrics

### 🔍 Correlation Analysis
- Identify relationships between metrics
- Understand performance drivers
- Optimize content strategy

## 🛠️ Troubleshooting

### Common Issues:

**1. Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

**2. Data Loading Issues**
- Check CSV file format and column names
- Ensure dates are in YYYY-MM-DD format
- Verify file paths in config.ini

**3. Dashboard Not Loading**
- Check if ports 8501/8050 are available
- Try different ports in config.ini
- Restart the dashboard application

**4. Chart Display Issues**
- Update your browser
- Clear browser cache
- Check Plotly version: `pip install --upgrade plotly`

## 📚 Dependencies

### Core Libraries:
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models

### Dashboard Libraries:
- **streamlit**: Web dashboard framework
- **dash**: Professional dashboard framework

### Export Libraries:
- **openpyxl**: Excel file generation
- **reportlab**: PDF report generation

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs or request features on GitHub
- **Documentation**: Check the wiki for detailed guides
- **Community**: Join discussions in the Issues section

## 🔮 Roadmap

### Upcoming Features:
- [ ] **Real-time API Integration** with YouTube Data API
- [ ] **Advanced ML Models** (Random Forest, Neural Networks)
- [ ] **Automated Insights** with AI-generated recommendations
- [ ] **Team Collaboration** features for multiple users
- [ ] **Mobile App** for on-the-go analytics
- [ ] **Custom Alerts** for performance thresholds
- [ ] **A/B Testing Tools** for content optimization
- [ ] **Competitive Analysis** features

## 📞 Contact

- **GitHub**: [@DuongCaoNhan](https://github.com/DuongCaoNhan)
- **Email**: duongcaonhan@example.com
- **LinkedIn**: [Duong Cao Nhan](https://linkedin.com/in/duongcaonhan)

---

⭐ **Star this repository** if you find it helpful!

🚀 **Happy analyzing your YouTube data!** 📺✨
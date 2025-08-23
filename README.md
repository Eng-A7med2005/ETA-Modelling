# TechnoColabs Delivery AI - ETA Prediction System

Advanced AI-powered delivery ETA prediction system with 99.96% accuracy.

## 🚀 Deployment on Render

### Prerequisites
- Render account
- Git repository with your code

### Quick Deploy on Render

1. **Fork/Clone this repository**
2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Sign up/Login
   - Click "New +" → "Web Service"

3. **Configure the service:**
   - **Name:** `technocolabs-delivery-ai`
   - **Environment:** `Docker`
   - **Region:** Choose closest to you
   - **Branch:** `main`
   - **Root Directory:** Leave empty
   - **Build Command:** `docker build -t technocolabs-delivery-ai .`
   - **Start Command:** `docker run -p $PORT:5000 technocolabs-delivery-ai`

4. **Environment Variables:**
   - `PYTHON_VERSION`: `3.9`
   - `PORT`: `5000`

5. **Click "Create Web Service"**

### Manual Deploy Steps

1. **Push your code to GitHub**
2. **Connect your repository to Render**
3. **Configure as Docker service**
4. **Deploy!**

## 🐳 Local Docker Testing

### Prerequisites
- Docker installed
- Docker Compose installed

### Run Locally

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t technocolabs-delivery-ai .
docker run -p 5000:5000 technocolabs-delivery-ai
```

## 📁 Project Structure

```
vercel-deploy/
├── api/
│   ├── __init__.py
│   ├── index.py          # Main Flask app with full ML model
│   ├── wsgi.py           # WSGI entry point
│   └── requirements.txt  # Python dependencies
├── best_model.pkl        # Trained ML model (4.3MB)
├── scaler.pkl           # Feature scaler (1.9KB)
├── feature_columns.pkl  # Feature names (575B)
├── label_encoders.pkl   # Label encoders (7.0MB)
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Local testing
├── render.yaml          # Render configuration
├── start.sh             # Startup script
└── requirements.txt     # Root dependencies
```

## 🔧 API Endpoints

- `GET /` - Web UI interface
- `POST /predict` - Main prediction endpoint
- `GET /health` - Health check
- `GET /info` - Model information
- `GET /test` - Test prediction

## 🐛 Troubleshooting

### Common Issues:

1. **Model Loading Error:**
   - Ensure all `.pkl` files are in the root directory
   - Check file permissions in Docker

2. **Docker Build Error:**
   - Check Dockerfile syntax
   - Verify all files are copied correctly

3. **Render Deployment Error:**
   - Check build logs in Render dashboard
   - Verify environment variables

4. **Memory/Timeout Error:**
   - Model files are large, may need more resources
   - Increase timeout settings

### Debug Steps:

1. **Test Locally with Docker:**
   ```bash
   docker-compose up --build
   ```

2. **Check Render Logs:**
   - Go to your service in Render dashboard
   - Check "Logs" tab

3. **Verify Files:**
   ```bash
   ls -la *.pkl
   ```

## 📊 Model Information

- **Accuracy:** 99.96%
- **Features:** 67 advanced features
- **Cities:** 5 Chinese cities supported
- **MAE:** 0.66 hours
- **Model Size:** ~12MB total

## 🏙️ Supported Cities

- Chongqing (CQ)
- Shanghai (SH)
- Hangzhou (HZ)
- Jinan (JL)
- Yantai (YT)

## 🔗 Links

- **Web UI:** Your deployed URL
- **API Docs:** Your deployed URL + `/info`
- **Health Check:** Your deployed URL + `/health`
- **Test Endpoint:** Your deployed URL + `/test`

## 💡 Advantages of Render Deployment

- ✅ **Full ML Model Support** - No size limitations
- ✅ **Docker Support** - Consistent environment
- ✅ **Auto-scaling** - Handles traffic spikes
- ✅ **SSL/HTTPS** - Secure by default
- ✅ **Custom Domains** - Professional URLs
- ✅ **Monitoring** - Built-in analytics


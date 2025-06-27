# 🌱 Smart Crop Advisor

Smart Crop Advisor is a complete farming assistant app that helps farmers make better decisions using **Artificial Intelligence (AI)** and **real-time data**.

This project was developed for **AgriTech Hackathon 2025** and provides tools like:
- Crop disease detection from leaf images
- NDVI crop health check using satellite images
- Weather-based disease forecasting
- AI chatbot for farming questions
- Fertilizer and yield calculators
- Farmer diary and planning tools

All features are combined in a simple **web app interface** built with [Streamlit](https://streamlit.io/), which works in your browser (no need to build a full website or app manually).

---

## 💡 What Can This App Do?

| Feature                        | What It Does                                                                 |
|-------------------------------|------------------------------------------------------------------------------|
| **1. Crop Disease Detection** | Upload a leaf photo → AI tells you the disease and suggests a cure.         |
| **2. NDVI Health Analysis**   | Upload RED and NIR images → shows a color map of crop health.               |
| **3. Weather Forecasting**    | Enter your location → app predicts risk of disease based on weather.        |
| **4. AI Assistant**           | Ask farming questions → chatbot replies with helpful advice.                |
| **5. Calculators**            | Plan your farm with fertilizer, yield, and crop rotation tools.             |
| **6. Diary & Logs**           | Keep notes of what you do each day on the farm.                             |
| **7. MongoDB Integration**    | Keeps track of user activity and prediction stats in a secure database.     |

---

## 🛠️ Technologies Used

| Tool/Library        | Purpose                                                        |
|---------------------|----------------------------------------------------------------|
| Python              | Programming language used to build the app                     |
| TensorFlow / Keras  | Used to train and run the disease detection AI model           |
| Streamlit           | Creates the interactive web app interface                      |
| MongoDB             | Stores user data and app statistics securely                   |
| OpenWeatherMap API  | Gets real-time weather data for forecasting                    |
| RasterIO + NumPy    | Used for image processing in NDVI health analysis              |
| Matplotlib          | Generates NDVI heatmaps and visual charts                      |

---

## 📦 How to Run This Project (Even If You're a Beginner)

### Step 1: Clone the Project
Download the code from GitHub to your computer.

```bash
git clone https://github.com/Adeshkachare/Smart_Crop_Advisor
cd Smart_Crop_Advisor
```

Step 2: Set Up Python Environment
Make sure you have Python 3.8 or later installed.
---------------------------------------------------------------------------------

Then run:  
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
---------------------------------------------------------------------------------

Step 3: Install Required Libraries
pip install -r requirements.txt
---------------------------------------------------------------------------------

Step 4: Add Your Secrets (API Keys)
Create a file called .streamlit/secrets.toml inside the project folder:

[mongo]
username = "your_mongodb_username"
password = "your_mongodb_password"

[api]
openrouter_api_key = "add_key_here"
weather_api_key = "add_key_here"

Don’t have a MongoDB or OpenRouter account?
Create a free MongoDB account at https://www.mongodb.com
Get OpenRouter API key at https://openrouter.ai
---------------------------------------------------------------------------------

Step 5: Run the App
streamlit run app.py
It will open the app in your default web browser at http://localhost:8501
---------------------------------------------------------------------------------

📁 Folder Overview
Crop_Disease_Detector/
├── app.py                      # Main app file to run
|
├── class_indices.json          # AI class labels for disease prediction
|
├── class_names.json            # Full disease class names
|
├── disease_faq.json            # Cure tips for each disease
|
├── crop_disease_model_optimized.keras  # Trained CNN model
|
├── feedback.py                 # Feedback form component
|
├── requirements.txt            # All libraries needed
|
└── .streamlit/
    |
    └── secrets.toml            # Your private API keys (you create this)
---------------------------------------------------------------------------------

🎯 Project Goal
Make modern farming easier and smarter using AI, computer vision,
and real-time weather data — even in rural areas with simple devices.
---------------------------------------------------------------------------------

🔮 Future Improvements

Retrain the model to reduce overfitting and improve accuracy
Train and add modules for:
- Weed detection
- Disease forecasting
Voice command and multilingual support
Drone image integration
Better UI for mobile users
---------------------------------------------------------------------------------

✅ Contributions
This project was built for learning and demonstration purposes. 
You are welcome to fork, improve, or use this as a base for 
your own AgriTech solutions.
---------------------------------------------------------------------------------

📜 License
This project is free for personal and academic use. For commercial use,
please contact the author.
---------------------------------------------------------------------------------

🔗 Links

Live Demo: https://www.youtube.com/watch?v=E19AvVu8KFk
GitHub Repository: https://github.com/Adeshkachare/Smart_Crop_Advisor
---------------------------------------------------------------------------------

🙌 Thank You
Thanks to everyone who supported or reviewed this project. 
Feedback and ideas are always welcome!
---------------------------------------------------------------------------------







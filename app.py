# The updated Smart Crop Advisor Streamlit App with MongoDB Integration

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.io import MemoryFile
from datetime import datetime
from pymongo import MongoClient
from urllib.parse import quote_plus
from scipy.ndimage import gaussian_filter
import requests
from feedback import feedback_form  # ✅ Correct



# --- App Config ---
st.set_page_config(
    page_title="Smart Crop Advisor",
    layout="wide",
    page_icon="🎾"
)

# --- Theme Colors ---
AGRI_GREEN = "#4CAF50"
AGRI_BROWN = "#8D6E63"
AGRI_BLUE = "#2196F3"

# --- MongoDB Connection ---
@st.cache_resource
def get_db():
    username = quote_plus(st.secrets["mongo"]["username"])
    password = quote_plus(st.secrets["mongo"]["password"])
    mongo_uri = f"mongodb+srv://{username}:{password}@cluster0.vpzsezg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)
    return client["smart_crop_advisor"]

db = get_db()

# --- MongoDB Stat Update Helper ---
def update_stat(field_name, change=1):
    db["stats"].update_one({"_id": "dashboard"}, {"$inc": {field_name: change}}, upsert=True)

# --- Load Class Indices ---
with open("class_indices.json") as f:
    class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

# --- Load Class Names ---
with open("class_names.json") as f:
    CLASS_NAMES = json.load(f)

# --- Load Disease Knowledgebase ---
with open("disease_faq.json") as f:
    DISEASE_KNOWLEDGE = json.load(f)

# --- Load Model ---
model = tf.keras.models.load_model("crop_disease_model_optimized.keras")

# --- Image Preprocessing ---
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

# --- Predict Disease ---
def predict_disease(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[::-1][:5]
    top_preds = [(idx_to_class[i], predictions[i]) for i in top_indices]
    return top_preds


# --- Home Dashboard ---
def home_dashboard():
    st.title("🏠 Welcome to Smart Crop Advisor")
    st.markdown("### 📊 Dashboard Overview")

    stats = db["stats"].find_one({"_id": "dashboard"}) or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Diseases", str(stats.get("diseases_detected", 0)), "+")
    col2.metric("NDVI Analyses", str(stats.get("ndvi_analyses", 0)), "+")
    col3.metric("Forecasts Given", str(stats.get("forecasts_given", 0)), "+")

    st.markdown("---")
    st.markdown("### 🌦️ Weather & Disease Alerts")
    st.success("✅ Your crops are healthy today! 🌱")

    st.markdown("### 📌 Quick Tips")
    st.info("💧 Water tomato crops every 2–3 days in hot weather.")
    st.info("🧪 Use neem oil spray to control early-stage pests.")

    st.markdown("### 🫟 Navigate from the sidebar to begin.")


# --- Crop Disease Detector ---
def crop_disease_detection_module():
    st.header("🌿 Crop Disease Detection")
    st.write("Upload a leaf image to detect potential diseases.")

    uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf", use_container_width=True)

            # Predict disease
            top_preds = predict_disease(image)
            top_class, confidence = top_preds[0]

            st.markdown(f"### 🔔 **Predicted Disease: {top_class.replace('_', ' ')} ({confidence * 100:.2f}%)**")

            # Cure info from DISEASE_KNOWLEDGE
            cure_advice = DISEASE_KNOWLEDGE.get(top_class, {}).get("cure", "Cure info not available.")
            st.markdown(f"**Cure Advice:** {cure_advice}")

            # Show top 5 predictions
            st.markdown("---")
            st.markdown("### Other possible predictions:")
            for cls, prob in top_preds[1:]:
                st.markdown(f"🔔 {cls.replace('_', ' ')} ({prob * 100:.2f}%)")

            st.warning("⚠️ Note: Predictions may be inaccurate due to dataset overfitting. Work is ongoing to improve accuracy.")

            # Update database stats
            if confidence * 100 >= 60:
                update_stat("diseases_detected", 1)
            else:
                update_stat("diseases_detected", -1)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    def ai_assistant():
        import requests

        st.header("🤖 Farming Help Assistant")
        st.write("Ask any farming question and get advice from the AI assistant.")

        question = st.text_input("Type your question here:", key="ai_question_input")

        if st.button("Get AI Advice"):
            if not question.strip():
                st.warning("Please type a question before clicking 'Get AI Advice'.")
                return

            # ✅ Use token directly from secrets
            hf_token = st.secrets.get("hf_token") or st.secrets.get("api", {}).get("hf_token")
            if not hf_token:
                st.error("❌ Huggingface API token ('hf_token') not found in secrets!")
                return

            API_URL = "https://api-inference.huggingface.co/models/gpt2"  # ✅ Verified public model
            headers = {
                "Authorization": f"Bearer {hf_token}"
            }

            payload = {
                "inputs": question,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }

            with st.spinner("💬 Thinking..."):
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result[0].get("generated_text", "").replace(question, "").strip()
                    st.markdown("**AI Response:**")
                    st.info(generated_text)
                else:
                    st.error(f"❌ API error: {response.status_code}")
                    st.text(response.text)
def ai_assistant():
    import requests

    st.header("🤖 Farming Help Assistant")
    st.write("Ask any farming question and get advice from the AI assistant.")

    question = st.text_input("Type your question here:", key="ai_question_input")

    if st.button("Get AI Advice"):
        if not question.strip():
            st.warning("Please type a question before clicking 'Get AI Advice'.")
            return

        # ✅ Use token directly from secrets
        hf_token = st.secrets.get("hf_token") or st.secrets.get("api", {}).get("hf_token")
        if not hf_token:
            st.error("❌ Huggingface API token ('hf_token') not found in secrets!")
            return

        API_URL = "https://api-inference.huggingface.co/models/gpt2"  # ✅ Verified public model
        headers = {
            "Authorization": f"Bearer {hf_token}"
        }

        payload = {
            "inputs": question,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "do_sample": True
            }
        }

        with st.spinner("💬 Thinking..."):
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                generated_text = result[0].get("generated_text", "").replace(question, "").strip()
                st.markdown("**AI Response:**")
                st.info(generated_text)
            else:
                st.error(f"❌ API error: {response.status_code}")
                st.text(response.text)


# --- Crop Planner & Calculator ---
def crop_planner_calculator():
    st.header("😰 Crop Planner & Calculator")
    tabs = st.tabs(["🌿 Fertilizers & Pesticides", "🌾 Yield Calculator", "🔄 Crop Rotation", "🧠 Crop Recommendation"])

    with tabs[0]:
        st.subheader("🌿 Fertilizer & Pesticide Calculator")
        crop = st.selectbox("Select Crop", ["Wheat", "Rice", "Maize", "Tomato"])
        area = st.number_input("Enter Area (in acres)", min_value=0.0, step=0.1)
        if st.button("Calculate Fertilizers"):
            if crop and area:
                nitrogen = round(area * 50, 2)
                phosphorus = round(area * 40, 2)
                potassium = round(area * 30, 2)
                pesticide = round(area * 5, 2)
                st.success(f"Apply {nitrogen}kg Nitrogen, {phosphorus}kg Phosphorus, {potassium}kg Potassium, and {pesticide}L pesticide for {area} acres of {crop}.")

    with tabs[1]:
        st.subheader("🌾 Yield Calculator")
        crop = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize"], key="yield_crop")
        area = st.number_input("Total Area (in acres)", min_value=0.0, step=0.1, key="yield_area")
        yield_rate = st.number_input("Estimated Yield per Acre (in quintals)", min_value=0.0, step=0.1)
        if st.button("Calculate Total Yield"):
            total_yield = round(area * yield_rate, 2)
            st.success(f"Estimated Total Yield for {crop} = {total_yield} quintals")

    with tabs[2]:
        st.subheader("🔄 Crop Rotation Suggestion")
        previous_crop = st.selectbox("Previously Grown Crop", ["Wheat", "Rice", "Soybean", "Cotton"])
        rotation_map = {
            "Wheat": "Pulses or Legumes",
            "Rice": "Mustard or Barley",
            "Soybean": "Sorghum or Wheat",
            "Cotton": "Groundnut or Chickpea"
        }
        suggestion = rotation_map.get(previous_crop, "No suggestion available")
        st.info(f"🌾 Next recommended crop: **{suggestion}**")

    with tabs[3]:
        st.subheader("🧠 Crop Recommendation")
        soil_type = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Black"])
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
        recommendations = {
            ("Loamy", "Kharif"): "Maize, Rice",
            ("Clay", "Rabi"): "Wheat, Mustard",
            ("Sandy", "Zaid"): "Watermelon, Cucumber",
            ("Black", "Kharif"): "Cotton, Soybean"
        }
        key = (soil_type, season)
        crops = recommendations.get(key, "Consult local extension officer")
        st.info(f"✅ Suitable Crops: {crops}")

# --- Farmer Work Diary ---
def farmer_diary():
    st.title("📔 Farmer Work Diary")
    st.write("Keep a simple record of your daily farming activities here.")

    if "entries" not in st.session_state:
        st.session_state.entries = []

    with st.form("diary_form"):
        date = st.date_input("Date of activity")
        activity = st.text_area("What did you do today?")
        weather = st.text_input("Weather (e.g., Sunny, Rainy)")
        inputs = st.text_input("Fertilizers/Pesticides used (if any)")
        notes = st.text_area("Additional notes (optional)")

        if st.form_submit_button("Save entry"):
            if not activity.strip():
                st.warning("Please describe your activity before saving.")
            else:
                st.session_state.entries.append({
                    "date": str(date),
                    "activity": activity.strip(),
                    "weather": weather.strip(),
                    "inputs": inputs.strip(),
                    "notes": notes.strip(),
                })
                st.success("Your entry has been saved!")

    st.markdown("---")
    st.header("📋 Previous diary entries")

    if not st.session_state.entries:
        st.info("You have not added any diary entries yet.")
    else:
        for entry in reversed(st.session_state.entries):
            with st.expander(f"{entry['date']} — {entry['activity'][:40]}..."):
                st.write(f"**Activity:** {entry['activity']}")
                st.write(f"**Weather:** {entry['weather'] or 'N/A'}")
                st.write(f"**Inputs used:** {entry['inputs'] or 'None'}")
                st.write(f"**Notes:** {entry['notes'] or 'None'}")


# --- NDVI Analysis ---
def ndvi_analysis():
    st.header("🚁 NDVI Crop Health Check")
    st.write("Upload two pictures of your field to check how healthy your crops are.")
    st.write("One picture should be RED band, and the other NIR (Near Infrared) band.")

    red_band = st.file_uploader("🔴 Upload RED band image (tif, png, jpg)", type=["tif", "tiff", "png", "jpg"], key="red")
    nir_band = st.file_uploader("🌌 Upload NIR band image (tif, png, jpg)", type=["tif", "tiff", "png", "jpg"], key="nir")

    if red_band and nir_band:
        try:
            with MemoryFile(red_band.read()) as memfile_red:
                with memfile_red.open() as src_red:
                    red = src_red.read(1).astype(np.float32)

            with MemoryFile(nir_band.read()) as memfile_nir:
                with memfile_nir.open() as src_nir:
                    nir = src_nir.read(1).astype(np.float32)

            if red.shape != nir.shape:
                st.error("❌ RED and NIR images must be the same size. Please upload matching images.")
                return

            red /= red.max() if red.max() > 1 else 1
            nir /= nir.max() if nir.max() > 1 else 1

            denominator = nir + red
            denominator[denominator == 0] = 1e-8
            ndvi = (nir - red) / denominator

            ndvi_score = np.nanmean(ndvi[np.isfinite(ndvi)])
            ndvi_score = np.clip(ndvi_score, -1.0, 1.0)

            status = "🟢 Healthy Vegetation" if ndvi_score > 0.6 else \
                     "🟡 Moderate Stress" if ndvi_score > 0.3 else \
                     "🔴 Severe Stress"

            st.markdown(f"**NDVI Result:** {status} (Average NDVI: {ndvi_score:.2f})")

            st.subheader("🖼️ NDVI Map")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            cmap = plt.get_cmap("RdYlGn")
            masked_ndvi = np.ma.masked_invalid(ndvi)
            smoothed_ndvi = gaussian_filter(masked_ndvi, sigma=1)
            im = ax.imshow(smoothed_ndvi, cmap=cmap, vmin=-1, vmax=1)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("NDVI Value", fontsize=12)
            ax.set_title("NDVI Map", fontsize=13)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)

            if ndvi_score > 0.3:
                update_stat("ndvi_analyses", 1)
            else:
                update_stat("ndvi_analyses", -1)

        except Exception as e:
            st.error(f"Error processing images: {e}")
    else:
        st.info("Please upload both RED and NIR band images to perform NDVI analysis.")


# --- AI Weed Detector ---
def ai_weed_detector():
    st.header("🌱 Weed Detector")
    st.write("Upload a picture of your field to check for common weeds (model coming soon).")
    uploaded_file = st.file_uploader("📷 Upload Field Image", type=["jpg", "jpeg", "png"], key="weed_upload")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your field photo", use_column_width=True)
        st.warning("🚧 Weed detection is not ready yet. Please check back later!")

# --- Placeholder for Under Development Modules ---
def placeholder_module(module_name):
    st.title(f"{module_name}")
    st.info("🚧 This module is under development. Check back soon!")
    st.image("https://img.icons8.com/clouds/100/agriculture.png", width=120)
    st.markdown("We are working hard to bring this tool to life for better crop decisions. Stay tuned!")




# --- Disease Forecasting ---
def disease_forecasting():
    st.header("🌦️ Disease Risk Forecast")
    st.write("Enter your location and crop type to see the disease risk based on current weather.")

    location = st.text_input("📍 Your Location (e.g., Pune)", value="Pune")
    crop = st.selectbox("🌾 Your Crop", ["Wheat", "Rice", "Cotton", "Soybean", "Tomato", "Maize", "Chili"])

    if st.button("🔍 Show Disease Risk"):
        try:
            API_KEY = "9fce5a03645bce02c9ced76ccc203f49"
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
            res = requests.get(url)

            if res.status_code != 200:
                st.error(f"❌ Could not get weather for '{location}'. Please check the location name.")
                return

            data = res.json()
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            rainfall = data.get("rain", {}).get("1h", 0.0)

            st.success(f"Weather in {location.title()}:")
            st.write(f"- 🌡️ Temperature: **{temp}°C**")
            st.write(f"- 💧 Humidity: **{humidity}%**")
            st.write(f"- 🌧️ Rainfall (last 1 hour): **{rainfall} mm**")

            if humidity > 80 and temp > 20 and rainfall > 0:
                risk = "🔴 High Risk of Disease (fungal or bacterial infections likely)"
            elif humidity > 60:
                risk = "🟡 Moderate Risk — keep an eye on your crops"
            else:
                risk = "🟢 Low Risk — good weather conditions"

            st.subheader(f"Disease Risk for {crop}:")
            st.info(risk)
            update_stat("forecasts_given", 1)

        except Exception as e:
            st.error(f"Error fetching data: {e}")



# --- Help & Tutorials ---
def help_and_tutorials():
    st.title("📚 Help & Tutorials")
    st.markdown("### 🎥 Video Guide")
    st.video("https://www.youtube.com/watch?v=lE3NPKvZ2Zs")

    st.markdown("### 📖 Text Guide")
    st.markdown("""
    - Go to **Crop Disease Detector** to upload an image.
    - Use **NDVI** to analyze crop health.
    - **Planner** suggests best crops and usage.
    - View diary and forecast regularly.
    """)

    st.markdown("### ❓ FAQs")
    st.markdown("**Q:** Can I use this app offline?\n**A:** Some features like forecasts need the internet.")

# --- Placeholder for Under Development Modules ---
def placeholder_module(module_name):
    st.title(f"{module_name}")
    st.info("🚧 This module is under development. Check back soon!")
    st.image("https://img.icons8.com/clouds/100/agriculture.png", width=120)
    st.markdown("We are working hard to bring this tool to life for better crop decisions. Stay tuned!")

# --- Sidebar Navigation ---
try:
    st.sidebar.image("https://img.icons8.com/ios-filled/100/4CAF50/farm-2.png", width=100)
except:
    st.sidebar.image("fallback_icon.png", width=100)

# --- Initialize selected module if not already ---
if "selected_module" not in st.session_state:
    st.session_state["selected_module"] = "🏠 Home Dashboard"

# --- Main Menu Options ---
menu_options = [
    "🏠 Home Dashboard",
    "🌿 Crop Disease Detector",
    "🚐 NDVI Analysis",
    "🌱 AI Weed Detector",
    "📈 Disease Forecasting",
    "🫐 AI Assistant",
    "😰 Crop Planner & Calculator",
    "📒 Farmer Diary",
    "📚 Help & Tutorials",
    "📬 Give Feedback"
]

# --- Sidebar Radio Selector (with unique key) ---
selected = st.sidebar.radio(
    "📋 Main Menu",
    menu_options,
    index=menu_options.index(st.session_state["selected_module"]),
    key="menu_selector_main"
)

# --- Update session state on selection ---
if selected != st.session_state["selected_module"]:
    st.session_state["selected_module"] = selected
    st.rerun()

# --- Extra Sidebar Info ---
st.sidebar.caption("🔗 Internet required for weather, AI, and updates.")
st.sidebar.caption("✅ Optimized for Mobile View")
st.sidebar.markdown("🎧 Voice Commands & Text-to-Speech Coming Soon")
st.sidebar.markdown("---")

# --- Optional Go to Home Button ---
if st.sidebar.button("🏠 Go to Home Dashboard"):
    st.session_state["selected_module"] = "🏠 Home Dashboard"
    st.rerun()

# --- Routing Based on Selection ---
selected_module = st.session_state["selected_module"]

if selected_module == "🏠 Home Dashboard":
    home_dashboard()
elif selected_module == "🌿 Crop Disease Detector":
    crop_disease_detection_module()
elif selected_module == "🚐 NDVI Analysis":
    ndvi_analysis()
elif selected_module == "🌱 AI Weed Detector":
    ai_weed_detector()
elif selected_module == "📈 Disease Forecasting":
    disease_forecasting()
elif selected_module == "🫐 AI Assistant":
    ai_assistant()
elif selected_module == "😰 Crop Planner & Calculator":
    crop_planner_calculator()
elif selected_module == "📒 Farmer Diary":
    farmer_diary()
elif selected_module == "📚 Help & Tutorials":
    help_and_tutorials()
elif selected_module == "📬 Give Feedback":
    feedback_form()
else:
    placeholder_module(selected_module)

# --- Dynamic CSS Styling ---
AGRI_GREEN = "#4CAF50"
bg_color = "#1e1e1e"
text_color = "#ffffff"
input_bg = "#333333"
input_text = "white"

# --- Highlight Active Tab (simple effect) ---
highlight_css = f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-size: 1.1rem;
    }}
    .stButton > button {{
        background-color: {AGRI_GREEN};
        color: white;
        border-radius: 8px;
        font-size: 1rem;
    }}
    input, textarea {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
    }}
    div[data-testid="stRadio"] label {{
        font-size: 1rem;
        padding: 6px 12px;
        margin-bottom: 4px;
        border-radius: 6px;
    }}
    div[data-testid="stRadio"] label[data-selected="true"] {{
        background-color: {AGRI_GREEN};
        color: white !important;
        font-weight: bold;
    }}
    </style>
"""
st.markdown(highlight_css, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Smart Crop Advisor © 2025 | Built for Farmers")

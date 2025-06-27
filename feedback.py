import streamlit as st
from datetime import datetime
import pandas as pd
import base64
import os
import smtplib
from email.message import EmailMessage

# --- CSV File Path ---
CSV_FILE = "feedback.csv"
SCREENSHOT_DIR = "screenshots"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# --- Email Acknowledgment Function ---
def send_ack_email(recipient_email, user_name):
    try:
        msg = EmailMessage()
        msg["Subject"] = "Thank You for Your Feedback!"
        msg["From"] = "youremail@gmail.com"  # Your Gmail
        msg["To"] = recipient_email
        msg.set_content(
            f"Hi {user_name},\n\nThank you for sharing your valuable feedback with us!\n\n- Smart Crop Advisor Team"
        )

        # Use Gmail SMTP server
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login("your_email", "your_password")  
        server.send_message(msg)
        server.quit()
    except Exception as e:
        st.warning(f"⚠️ Could not send email: {e}")

# --- Feedback Form ---
def feedback_form():
    st.title("📬 Give Your Feedback")
    st.markdown("We'd love to hear your thoughts! Your feedback helps us grow. 🌾")

    with st.form("feedback_form"):
        name = st.text_input("👤 Your Name")
        contact = st.text_input("📱 Contact Number")
        gmail = st.text_input("📧 Gmail Address (optional, to receive updates)")
        rating = st.slider("⭐ Rate Your Experience", 1, 5, 3)
        recommend = st.radio("🤔 Would You Recommend This App?", ["Yes", "Maybe", "Not yet"])
        suggestion = st.text_area("💡 Suggestions / Feedback")
        screenshot = st.file_uploader("🖼️ Upload Screenshot (Optional)", type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        if not name.strip() or not contact.strip():
            st.warning("⚠️ Please fill in your name and contact details.")
            return

        image_path = ""
        if screenshot:
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            image_path = os.path.join(SCREENSHOT_DIR, f"{name}_{timestamp_str}_{screenshot.name}")
            with open(image_path, "wb") as f:
                f.write(screenshot.read())

        feedback_entry = {
            "Timestamp": datetime.now().isoformat(),
            "Name": name.strip(),
            "Contact": contact.strip(),
            "Gmail": gmail.strip(),
            "Rating": rating,
            "Recommend": recommend,
            "Suggestion": suggestion.strip(),
            "Screenshot": image_path
        }

        # Save to CSV
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            df = pd.concat([df, pd.DataFrame([feedback_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([feedback_entry])

        df.to_csv(CSV_FILE, index=False)
        st.success("✅ Thank you! Your feedback has been recorded.")

        if gmail:
            send_ack_email(gmail.strip(), name.strip())

# --- Run the App ---
if __name__ == "__main__":
    feedback_form()

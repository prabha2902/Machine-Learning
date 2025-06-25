# Machine-Learning
Fraud spotting and precaution by using machine learning random forest algorithm
**Run the App**
      streamlit run app.py
**🧪 How to Use**
1. Launch the app
2. Enter transaction details 
3. Click "Check Transaction"
       If it's fraud:
          You will see a red warning
       An email alert will be sent to the registered email
       If it's legitimate:
           You will see a green success message
****✉️ Email Notification Setup**
Edit the send_email_alert() function with:
SENDER = "your-email@gmail.com"
APP_PASSWORD = "your-app-password"  # Use App Password (not your Gmail password)
TO_EMAIL = "user@example.com"


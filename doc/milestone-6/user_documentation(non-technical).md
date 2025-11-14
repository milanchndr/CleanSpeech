# **A User's Guide to CleanSpeech**

Welcome to CleanSpeech! This guide will help you understand what CleanSpeech is, how to use it, and what to do if you run into any issues.

## **1. What is CleanSpeech?** 💬

CleanSpeech is an AI-powered communication assistant designed to help you write more clearly and respectfully online. Think of it as a spell-checker, but for politeness and tone.

You can use it to:
*   **Check your own comments** before posting them on social media, forums, or in games.
*   **Understand why** a comment might be considered toxic by an automated system.
*   **Get suggestions** on how to rephrase your message to be more constructive while keeping your original meaning.

Our goal is to help build healthier online communities by making moderation transparent and educational.

## **2. How to Use CleanSpeech: A Step-by-Step Guide**

Using the app is simple! Just follow these steps.

### **Step 1: Open the App**

First, you need to launch the application.

```bash
# 1. Open your computer's terminal or command prompt.
# 2. Navigate to the CleanSpeech project folder using the 'cd' command.
# Example: cd path/to/your/CleanSpeech_folder

# 3. Run the following command:
streamlit run ui/app.py
```
This will automatically open the CleanSpeech application in a new tab in your web browser.

### **Step 2: Go to the "Chat" Page**

The application has two main pages, which you can select from the navigation bar:
*   **💬 Chat:** This is where you analyze your text.
*   **🔍 Explain:** This gives you a deep dive into the AI's reasoning.

Start on the **Chat** page.

### **Step 3: Enter Your Text**

Type or paste the text you want to analyze into the large text box labeled **"Enter your text:"**.

`[SCREENSHOT: A view of the 'Chat' page with the text input box highlighted.]`

### **Step 4: Analyze It!**

Click the **"Predict"** button. In a few moments, CleanSpeech will analyze your text and show you the results.

### **Step 5: Understand the Results**

After you click predict, you'll see a few things:

**A. The Overall Verdict:**
A colored banner will tell you if your text was flagged for any type of toxicity.

`[SCREENSHOT: The red "Flagged as toxic" banner that appears after analysis.]`

**B. The Toxicity Breakdown:**
You'll see a bar chart showing the AI's confidence level across six categories: `toxic`, `insult`, `obscene`, `threat`, `severe_toxic`, and `identity_hate`. A longer bar means the AI is more certain the comment fits that category.

`[SCREENSHOT: The horizontal bar chart showing the probability scores for each of the six toxicity labels.]`

**C. The Constructive Suggestion (If Needed):**
If your comment was flagged, CleanSpeech will offer a rewritten, more constructive version. This suggestion aims to keep your original point but phrase it more respectfully.

`[SCREENSHOT: An example showing a toxic comment and the "Rewritten Text" suggestion below it.]`

### **Step 6 (Optional): Dive Deeper with "Explain"**

If you're curious about *why* your text was flagged, navigate to the **🔍 Explain** page. Here, you'll see a detailed breakdown of the last comment you analyzed.

*   **Color-Coded Words:** The most important feature! Words that increased the toxicity score are highlighted in **<font color="red">red</font>**, and words that decreased it are highlighted in **<font color="blue">blue</font>**.
*   **Word Importance Chart:** A bar chart that ranks the most impactful words in your comment.
*   **Attention Heatmap:** A visual grid that shows which words the AI focused on the most when making its decision.

`[SCREENSHOT: The 'Explain' page showing the color-coded text for "You are an idiot" with "idiot" highlighted in red.]`

### **Step 7 (Optional): Adjust the Sensitivity**

On the left-hand sidebar, you'll find **Toxicity Threshold** sliders. You can move these sliders to make CleanSpeech more or less strict for each category.

*   **Move a slider to the left** to make the AI less sensitive (it will flag fewer comments).
*   **Move a slider to the right** to make it more sensitive (it will flag more comments).

Click the **"Submit"** button in the sidebar to apply your changes.

`[SCREENSHOT: The sidebar showing the six sliders for adjusting the toxicity thresholds.]`

## **3. Example Texts to Try**

Here are a few examples you can copy and paste to see how CleanSpeech works.

*   **Clearly Toxic:** `You are an idiot and your idea is trash.`
    *   *Expected Result:* High scores for `toxic` and `insult`. You'll get a clear explanation and a constructive rewrite.

*   **Non-Toxic Disagreement:** `I disagree with your point, but I see where you're coming from.`
    *   *Expected Result:* Very low scores. The app will confirm it's not toxic and won't suggest a rewrite.

*   **Example with Negation:** `You are not a bad person.`
    *   *Expected Result:* On the "Explain" page, you will see the word **"not"** highlighted in **<font color="blue">blue</font>**, showing that it actively reduced the toxicity score.

## **4. Troubleshooting**

If something isn't working right, here are a few common solutions.

*   **The app doesn't load or is slow.**
    *   Make sure you have a stable internet connection.
    *   Try refreshing the browser page.
    *   If you are running it locally, check the terminal for any error messages.

*   **I get an error message after clicking "Predict".**
    *   The service might be temporarily busy. Please wait a moment and try again.
    *   Your text might be too long. Try analyzing a shorter piece of text.

*   **The AI's analysis seems wrong.**
    *   Artificial intelligence isn't perfect and can sometimes make mistakes, especially with sarcasm or complex context.
    *   Try adjusting the **sensitivity sliders** in the sidebar to better match your expectations.

---

Thank you for using CleanSpeech! We hope it helps you navigate online conversations more effectively.

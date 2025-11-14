# **CleanSpeech: A Guide for Building Healthier Online Communities**

Welcome to CleanSpeech! This guide will walk you through how to use our tool to understand, manage, and reduce toxicity in your online platform.

## 1. What is CleanSpeech?

CleanSpeech is more than just a tool that flags bad comments. It's an intelligent system designed to help you build a safer and more constructive online environment.

It helps you in three key ways:

*   ✅ **Detect:** It automatically identifies not just *if* a comment is toxic, but also *what kind* of toxicity it contains (e.g., insult, threat, hate speech).
*   🤔 **Explain:** It shows you exactly **which words** in a comment made it toxic, taking the guesswork out of moderation.
*   💡 **Improve:** It provides a **constructive rewrite** of toxic comments, showing users how they could have expressed their opinion without breaking community guidelines.

This approach helps you make fair, transparent moderation decisions and educates your users on how to communicate better.

## 2. Who is This For?

CleanSpeech is designed for anyone responsible for maintaining the health and safety of an online community, including:
*   **Community Managers**
*   **Social Media Moderators**
*   **Forum Administrators**
*   **Online Gaming Moderators**
*   **Customer Support Teams**

## 3. How to Use CleanSpeech: A Step-by-Step Guide

### Step 1: Open the CleanSpeech App

Getting started with CleanSpeech is easy. There's no software to install on your computer because the application is hosted online and ready to use.

1.  Open your preferred web browser (like Chrome, Firefox, or Safari).
2.  Navigate to the following web address to open the app:

    **[https://cleanspeech.streamlit.app/](https://cleanspeech.streamlit.app/)**

The CleanSpeech interface will load directly in your browser, and you'll be ready to start analyzing text.
---

### Step 2: Analyze Your Text

The main screen is your analysis hub. It’s designed to be simple and intuitive.

1.  **Enter Text:** Copy and paste or type the comment you want to analyze into the text box labeled "**Enter your text**".
2.  **Click Predict:** Hit the "**Predict**" button to start the analysis.

*   `[SCREENSHOT: The main "Chat" page of the Streamlit UI, showing the text input box with an example comment like "You are an idiot. This is the worst thing ever." and the "Predict" button.]`

---

### Step 3: Understand the Results

After a few moments, the system will display a full breakdown of the comment. Here’s what each section means:

#### **A. Overall Toxicity Score**

This section gives you a quick, at-a-glance view of the analysis.

*   **Flagged as toxic:** A red bar at the top tells you the comment has crossed the toxicity threshold.
*   **Probability Bars:** These bars show the system's confidence level (from 0% to 100%) for each of the six toxicity categories. A longer bar means a higher probability.

#### **B. The Explanation: Why Was It Flagged?**

This is where CleanSpeech shines. It highlights the specific words that contributed to the toxicity score.

*   <span style="background-color:#ffcccb; padding:2px; border-radius:3px;">Words in red</span> are the primary reason the comment was flagged as toxic.
*   <span style="background-color:#add8e6; padding:2px; border-radius:3px;">Words in blue</span> actually *decreased* the toxicity score (e.g., words like "not" or "don't").
*   <span style="color:grey;">Grey words</span> were neutral.

#### **C. The Constructive Rewrite**

If a comment was flagged as toxic, CleanSpeech provides a rewritten version. This version keeps the original user's core opinion but phrases it in a respectful and constructive way. This is a powerful tool for providing feedback to users.

*   `[SCREENSHOT: A complete analysis result. It should show the "Flagged as toxic" banner, the probability bars, the color-coded text explanation, and the "Constructive Rewrite" section.]`

---

### Step 4: Customize the Sensitivity (Optional)

Every community has different standards. You can fine-tune CleanSpeech's sensitivity using the sliders in the left-hand sidebar.

*   **Adjusting Thresholds:** A threshold is the "tipping point" for a comment to be flagged.
    *   **Lowering the threshold** (e.g., from 0.8 to 0.6) makes the system **more sensitive** and more likely to flag comments.
    *   **Increasing the threshold** makes the system **less sensitive** and will only flag more severe comments.

You can set a unique threshold for each of the six toxicity categories to match your community's specific guidelines.

*   `[SCREENSHOT: The sidebar of the UI, clearly showing the six toxicity threshold sliders.]`

## 4. Example Scenarios

#### Scenario 1: A Clear Insult

*   **Input:** `"You are a moron for saying that."`
*   **Expected Output:**
    *   High probability for `insult` and `toxic`.
    *   The words `"moron"` will be highlighted in <span style="background-color:#ffcccb; padding:2px; border-radius:3px;">red</span>.
    *   **Rewrite:** `"I completely disagree with that statement and find it unconvincing."`

#### Scenario 2: A Non-Toxic Disagreement

*   **Input:** `"I don't agree with your analysis, but I appreciate the detailed post."`
*   **Expected Output:**
    *   Low probabilities across all categories.
    *   The comment will be flagged as **non-toxic**.
    *   No rewrite will be provided.

## 5. Troubleshooting

Having trouble? Here are a few common issues and their solutions.

*   **The App Doesn't Load in My Browser:**
    *   Make sure you ran the `streamlit run ui/app.py` command correctly from the project directory. Check for any error messages in your terminal.

*   **The Analysis is Slow:**
    *   The first analysis after launching the app can be slower as the model loads. Subsequent analyses should be faster. Complex, long comments also take more time to process.

*   **A Comment Was Flagged That I Think is Fine (False Positive):**
    *   This is a great time to use the threshold sliders! Try increasing the threshold for the specific category that was flagged (e.g., increase the `insult` threshold). This makes the system less sensitive to borderline comments.

*   **I See an Error Message:**
    *   The simplest first step is to refresh your browser page. If that doesn't work, try stopping the app in your terminal (press `Ctrl + C`) and running the launch command again.

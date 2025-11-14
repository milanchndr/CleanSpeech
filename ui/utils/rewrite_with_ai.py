from _temp.config import THRESHOLD, GEMINI_MODEL
import google.generativeai as genai

from dotenv import main
import os 
_ = main.load_dotenv(main.find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
def rewrite_with_gemini(text, toxicity_dict):
    """
    Convert toxic comments into constructive criticism using Gemini.
    Preserves the core message while removing offensive language.
    If all toxicity scores are below threshold, return original unchanged.
    """
    # Check if ALL toxicity scores are below threshold
    max_tox = max(toxicity_dict.values())
    if max_tox < THRESHOLD:
        print(f"   ℹ️  All toxicity scores below {THRESHOLD*100:.0f}%, no rewrite needed")
        return text

    # Format toxicity scores as percentages for the prompt
    tox_breakdown = "\n".join([f"  - {label}: {prob*100:.1f}%" for label, prob in toxicity_dict.items()])

    # Few-shot prompt with examples
    prompt = f"""You are a communication assistant that transforms toxic comments into constructive criticism. Your goal is to preserve the core message, opinion, and intent while removing offensive language and making it constructive.

TOXICITY ANALYSIS:
{tox_breakdown}

EXAMPLES OF GOOD TRANSFORMATIONS:

Example 1:
Toxic: "You are such an idiot and I hate you!"
Constructive: "I strongly disagree with your approach and find it frustrating."

Example 2:
Toxic: "This movie is garbage, only idiots would like it."
Constructive: "I found this movie disappointing and don't understand its appeal."

Example 3:
Toxic: "That guy is a complete moron."
Constructive: "I think that person made some poor decisions."

Example 4:
Toxic: "This politician is a liar and a thief."
Constructive: "I question this politician's honesty and integrity based on their actions."

Example 5:
Toxic: "I can't stand that disgusting singer."
Constructive: "I really dislike that singer's style and find it unappealing."

KEY PRINCIPLES:
- Keep the underlying opinion, critique, or sentiment
- Remove insults, slurs, and aggressive language
- Make it constructive and respectful
- Don't dilute the message - if someone is angry, show disagreement/frustration
- Don't lose the specificity of what they're criticizing
- Maintain the emotional intensity in a constructive way

Now transform this comment:
Original: "{text}"

Provide ONLY the constructive version, no explanations."""

    try:
        response = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
        rewritten = response.text.strip() if response.text else text
        return rewritten
    except Exception as e:
        print(f"   ⚠️  Gemini API error: {e}")
        return text

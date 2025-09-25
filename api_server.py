from flask import Flask, request, jsonify
from inference import classify_intent, generate_response_gpt, find_similar_question, summarize_text

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    intent = classify_intent(text)
    gpt_response = generate_response_gpt(text)
    similar_faq = find_similar_question(text, ["How to block a lost card?", "How to reset my password?"])
    summary = summarize_text(text)

    return jsonify({
        "intent": intent,
        "gpt_response": gpt_response,
        "similar_faq": similar_faq,
        "summary": summary
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)

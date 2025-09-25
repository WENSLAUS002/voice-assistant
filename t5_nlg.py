from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5NLG:
    def __init__(self, model_name="t5-small"):
        """Initialize the T5 model and tokenizer."""
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate_text(self, task_prefix, input_text, max_length=50):
        """
        Generate text using the T5 model.

        :param task_prefix: The prefix for the task (e.g., 'summarize:', 'question:', etc.)
        :param input_text: The input text for processing.
        :param max_length: The maximum length of the generated output.
        :return: The generated text.
        """
        input_text = f"{task_prefix} {input_text}".strip()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output_ids = self.model.generate(input_ids, max_length=max_length)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

if __name__ == "__main__":
    t5_nlg = T5NLG(model_name="t5-small")

    # Example 1: Text Summarization
    input_text = "The banking system ensures secure transactions and customer support."
    summary = t5_nlg.generate_text("summarize:", input_text)
    print("Summary:", summary)

    # Example 2: Question Answering using T5 format
    question = "Who ensures secure transactions?"
    context = "The banking system ensures secure transactions and customer support."
    qa_input = f"question: {question} context: {context}"
    answer = t5_nlg.generate_text("", qa_input)
    print("Answer:", answer)

    # Example 3: Paraphrasing (if supported by your fine-tuned model)
    sentence = "The banking system ensures secure transactions and customer support."
    paraphrased = t5_nlg.generate_text("paraphrase:", sentence)
    print("Paraphrased:", paraphrased)

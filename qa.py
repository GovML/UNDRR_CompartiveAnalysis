from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def load_qa_model(model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

def load_ranking_model(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def split_text_with_overlap(text, max_length=1024, overlap=200):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        start += max_length - overlap
    return chunks

def predict_answers(model, tokenizer, question, context_chunks):
    answers = []
    for chunk in context_chunks:
        inputs = tokenizer(question, chunk, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_scores, end_scores = outputs.start_logits, outputs.end_logits
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))
        answers.append(answer)
    return answers

def rerank_answers(ranking_model, ranking_tokenizer, question, answers):
    scores = []
    for answer in answers:
        inputs = ranking_tokenizer(question, answer, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            score = ranking_model(**inputs).logits.squeeze().item()
        scores.append(score)
    
    ranked_answers = [ans for _, ans in sorted(zip(scores, answers), reverse=True)][:3]
    return ranked_answers

def run_qa_on_text(file_path, question):
    with open(file_path, "r", encoding="utf-8") as file:
        context = file.read()
    
    tokenizer, model = load_qa_model()
    ranking_tokenizer, ranking_model = load_ranking_model()
    
    context_chunks = split_text_with_overlap(context)
    answers = predict_answers(model, tokenizer, question, context_chunks)
    top_answers = rerank_answers(ranking_model, ranking_tokenizer, question, answers)
    
    print("Top 3 Answers:")
    for i, answer in enumerate(top_answers, 1):
        print(f"{i}. {answer}")

if __name__ == "__main__":
    file_path = "C:/Users/Rudy/Desktop/UN-run/undrr/action_plan.txt"  # Change this to the actual text file path
    question = "What institutional mechanisms are in place to monitor, evaluate, and ensure accountability for the DRR strategy’s implementation?"
    run_qa_on_text(file_path, question)

from qa.deepseek_client import DeepSeekClient 

class AlignmentScorer:
    def __init__(self, model_name="deepseek-r1:7b", temperature=0.6):
        self.client = DeepSeekClient(model_name=model_name, temperature=temperature)

    def score_alignment(self, source: str, target: str) -> str:
        prompt = f"""You are an evaluator tasked with scoring how well one text (the response) aligns with another (the reference). 
Give a score from 1 to 10, where:
- 10 means perfect alignment (completely accurate and faithful),
- 5 means partial alignment (some relevance or correctness but with issues),
- 1 means completely unrelated or incorrect.

Then briefly explain the reasoning.

Reference:
{source}

Response:
{target}

Score and explanation:"""

        response = self.client.generate(prompt)
        return response['text'] if isinstance(response, dict) and 'text' in response else str(response)

# Example
if __name__ == "__main__":
    scorer = AlignmentScorer()
    reference = "The cat sat on the mat and looked out the window."
    response = "The feline rested on the carpet, gazing outside."
    print(scorer.score_alignment(reference, response))
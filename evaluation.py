from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline

class TextComparison:
    def __init__(self, file1_path, file2_path):
        self.list1 = self.load_file(file1_path)
        self.list2 = self.load_file(file2_path)
        self.pico_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.pico_labels = ["Population", "Intervention", "Comparison", "Outcome"]

    def load_file(self, file_path):
        """
        Load a list of lines from a text file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def calculate_rouge_1(self, ref, hyp):
        """
        Calculate ROUGE 1 score.
        """
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        ref_counter = Counter(ref_tokens)
        hyp_counter = Counter(hyp_tokens)
        overlap = sum((ref_counter & hyp_counter).values())
        return overlap / len(ref_tokens)

    def calculate_rouge_l(self, ref, hyp):
        """
        Calculate ROUGE-L score using LCS.
        """
        def lcs(a, b):
            lengths = [[0 for _ in range(len(b)+1)] for _ in range(len(a)+1)]
            for i, x in enumerate(a):
                for j, y in enumerate(b):
                    if x == y:
                        lengths[i+1][j+1] = lengths[i][j] + 1
                    else:
                        lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
            return lengths[len(a)][len(b)]

        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        lcs_length = lcs(ref_tokens, hyp_tokens)
        return lcs_length / len(ref_tokens)

    def calculate_glue_similarity(self, ref, hyp):
        """
        Calculate a simple similarity score as a placeholder for GLUE.
        """
        def cosine_similarity(a, b):
            vectorizer = CountVectorizer().fit_transform([a, b])
            vectors = vectorizer.toarray()
            num = vectors[0] @ vectors[1]
            denom = (vectors[0]**2).sum()**0.5 * (vectors[1]**2).sum()**0.5
            return num / denom

        return cosine_similarity(ref, hyp)

    def calculate_pico(self, text):
        """
        Evaluate PICO elements using zero-shot classification.
        """
        pico_score = self.pico_model(text, self.pico_labels)
        return {label: score for label, score in zip(pico_score['labels'], pico_score['scores'])}

    def compare_texts(self):
        """
        Compare each line from the two lists.
        """
        results = []
        for ref, hyp in zip(self.list1, self.list2):
            rouge_1 = self.calculate_rouge_1(ref, hyp)
            rouge_l = self.calculate_rouge_l(ref, hyp)
            glue_sim = self.calculate_glue_similarity(ref, hyp)
            pico_ref = self.calculate_pico(ref)
            pico_hyp = self.calculate_pico(hyp)
            results.append({
                "ROUGE-1": rouge_1,
                "ROUGE-L": rouge_l,
                "GLUE-like Similarity": glue_sim,
                "PICO (Reference)": pico_ref,
                "PICO (Hypothesis)": pico_hyp
            })
        return results

file1_path = "data/sonder_human_answers.txt" 
file2_path = "data/sonder_llm_answers.txt"

text_comparison = TextComparison(file1_path, file2_path)
comparison_results = text_comparison.compare_texts()

for i, result in enumerate(comparison_results):
    print(f"Comparison {i+1}: {result}")
    print("\n")

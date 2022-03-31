if __name__ == "__main__":
    import sys
    sys.path.append("..")

from sentence_transformers import SentenceTransformer
from util.nlp import dot_score

def test_subject_comparison():
    # TODO: Fix this to be more you know
    generated_questions = ["Why are your titties so big and your ass so fat?", 
                    "How do you like it when I take a casual stroll through the park?", 
                    "Why are men like this?"]
    subjects = ["asses", "titties", "walking", "nature", "parks"]


    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    question_embs = [model.encode(gq) for gq in generated_questions]
    subj_embs = [model.encode(keyphrase) for keyphrase in subjects]

    # Get the best score for each question from all subject pairings
    best_score = [0.0] * len(question_embs)
    related_subject = [None] * len(question_embs)
    for q_idx, q_emb in enumerate(question_embs):
        for s_idx, subj_emb in enumerate(subj_embs):
            # Calculate -- TODO maybe remove the loop over subject embeds
            score = dot_score(q_emb, subj_emb)[0].tolist()[0]
            if score > best_score[q_idx]:
                best_score[q_idx] = score
                related_subject[q_idx] = s_idx

    # indices to subject names
    related_subject = [subjects[subj_idx] for subj_idx in related_subject]
    question_subject_relevance = zip(generated_questions, related_subject, best_score)
    _ = [print(x) for x in question_subject_relevance]
    # for all subjects
    # dot score for question and subject qs
    # dot score for answer and subject as
    # keep highest score of qs, as
    # keep highest score for all subjects
    # rank qa 

if __name__ == "__main__":
    test_subject_comparison()
from similarity.text_similarity import TextSimilarity

# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
index = ['Health', 'Brocolli']
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

similarity = TextSimilarity(index=index, topics_number=5)
similarity.set_document_list(doc_set)
similarity.build_models()

sims_tfidf = similarity.calculate_pairwise_documents_similarity_tfidf()
sim_tfidf = similarity.similarity_tfidf(doc_a, doc_e)
for rec in sims_tfidf:
    print(rec)
print(sim_tfidf)

sims_lsi = similarity.calculate_pairwise_documents_similarity_lsi()
sim_lsi = similarity.similarity_lsi(doc_a, doc_e)
for rec in sims_lsi:
    print(rec)
print(sim_lsi)

sims_lda = similarity.calculate_pairwise_documents_similarity_lda()
sim_lda = similarity.similarity_lda(doc_a, doc_e)
for rec in sims_lda:
    print(rec)
print(sim_lda)
import matplotlib.pyplot as plt

# Visualizzare i topic in un grafico a barre
def plot_topics(topics, top_n=10):
    # Mostra solo i primi 10 topic
    for topic_id, topic_words in list(topics.items())[:10]:
        words = [word for word, _ in topic_words[:top_n]]
        freqs = [freq for _, freq in topic_words[:top_n]]
        
        plt.figure(figsize=(10, 5))
        plt.barh(words, freqs)
        plt.title(f'Topic {topic_id}')
        plt.xlabel('Peso')
        plt.ylabel('Parole')
        plt.gca().invert_yaxis()  # Inverte l'asse y per avere il topic più importante in alto
        plt.show()

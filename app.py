from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy.spatial.distance import cosine


app = Flask(__name__)

# Load the Universal Sentence Encoder model
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(model_url)

@app.route('/embeddings', methods=['GET'])
def get_embeddings():
    try:
        sentence = request.args.get('sentence')
        if sentence is None:
            return jsonify({"error": "Missing 'sentence' parameter"}), 400

        # Tokenize and embed the input sentence
        embedding = embed([sentence]).numpy()[0].tolist()
        return jsonify({"embedding": embedding})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/embeddings/bulk', methods=['POST'])
def get_bulk_embeddings():
    try:
        data = request.get_json()
        sentences = data.get('sentences',[])

        if not sentences:
            return jsonify({"error": "No sentences provided in the request"}), 400

        embeddings = embed(sentences)
        embeddings = embeddings.numpy().tolist()
            
        return jsonify({"embeddings": embeddings})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/embeddings/similarity', methods=['POST'])
def get_similarity():
    try:
        data = request.get_json()
        sentence_1 = data.get('sentence_1', '')
        sentence_2 = data.get('sentence_2', '')

        if not sentence_1 or not sentence_2:
            return jsonify({"error": "Both sentence_1 and sentence_2 are required in the request"}), 400

        # Tokenize and embed both sentences
        embedding_1 = embed([sentence_1]).numpy()[0]
        embedding_2 = embed([sentence_2]).numpy()[0]

        # Calculate the cosine similarity between the embeddings
        similarity_score = 1 - cosine(embedding_1, embedding_2)
        return jsonify({"similarity": round(similarity_score,2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stopServer', methods=['GET'])
def stopServer():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({ "success": True, "message": "Server is shutting down..." })

if __name__ == '__main__':
    app.run(host='localhost', port=5000)

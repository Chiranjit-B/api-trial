from flask import Flask, request, jsonify
import openai
import pickle
import faiss
import numpy as np
import os
from dotenv import load_dotenv
import logging

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")
openai.api_key = openai_api_key

# File paths - Update these paths to relative paths or ENV variables for production
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.index")
ID_MAPPING_PATH = os.getenv("ID_MAPPING_PATH", "id_mapping.pkl")
CONTENT_DICT_PATH = os.getenv("CONTENT_DICT_PATH", "content_dict.pkl")

# Load FAISS index, ID mapping, and content dictionary
try:
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    id_mapping = pickle.load(open(ID_MAPPING_PATH, "rb"))
    content_dict = pickle.load(open(CONTENT_DICT_PATH, "rb"))
except Exception as e:
    raise RuntimeError(f"Error loading FAISS index or data files: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_embedding_for_query(query):
    """Generate embedding for the user query using OpenAI API."""
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )
        return np.array(response["data"][0]["embedding"], dtype=np.float32)
    except Exception as e:
        logger.error(f"Error generating embedding for query: {e}")
        return None

@app.route('/rag', methods=['POST'])
def rag_api():
    try:
        # Parse input JSON
        data = request.get_json()
        user_query = data.get("query")
        top_k = data.get("top_k", 5)

        if not user_query:
            return jsonify({"error": "Query is required"}), 400

        # Step 1: Generate embedding for the user query
        query_embedding = generate_embedding_for_query(user_query)
        if query_embedding is None:
            return jsonify({"error": "Failed to generate query embedding"}), 500

        # Step 2: Perform semantic search
        distances, indices = faiss_index.search(query_embedding.reshape(1, -1), top_k)
        results = []
        context = ""

        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(id_mapping):
                result_id = id_mapping[idx] if isinstance(id_mapping, list) else id_mapping.get(idx)
                if result_id:
                    for item in content_dict:
                        if "id" in item and item["id"] == result_id:
                            file_name_chunk_no = item.get("text_chunk", "No file_name_chunk_no available")
                            results.append({"file_name_chunk_no": file_name_chunk_no, "distance": float(dist)})
                            context += f"{file_name_chunk_no}\n\n"
                            break

        # Step 3: Generate a response using RAG
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions using the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}"}
            ]
        )

        generated_response = response["choices"][0]["message"]["content"]

        return jsonify({
            "query": user_query,
            "results": results,
            "response": generated_response
        })

    except Exception as e:
        logger.error(f"Error in /rag endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use production-ready server
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)

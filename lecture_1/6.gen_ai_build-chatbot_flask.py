# Step 1: Import Flask tools 
from flask import Flask, request, render_template
from flask_cors import CORS
import json
# Step 2: Import our required tools from the transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

# Step 3: Choosing a model
# Open-source license and runs relatively fast.
model_name = "facebook/blenderbot-400M-distill"
 # Step 4: Fetch the model and initialize a tokenizer
 # Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Step 5: Chat
# Step 5.1: Keeping track of conversation history
conversation_history = []
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    # Read prompt from HTTP request body
    data = request.get_data(as_text=True)
    data = json.loads(data)
    print(data) # DEBUG
    input_text = data['prompt']
    # Create conversation history string
    # history = "\n".join(conversation_history)
    # Tokenize the input text and history
    #inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")
    # Encode the conversation history and user input    
    inputs = tokenizer(
        conversation_history + [input_text],  # keep it as a list for BlenderBot tokenizer
        return_tensors="pt",
        padding=True
        )
    
    # Generate the response from the model
    outputs = model.generate(**inputs,
                             max_new_tokens=50,   # Limit response length
                             temperature=0.7      #controlled randomness, so the model doesn’t repeat exactly.
                             )  # max_length will acuse model to crash at some point as history grows
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    app.run(debug=True)

## In orde rt test the curl to make a POST request:
## Open a new terminal: Select terminal tab –> open new terminal
# curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you today?"}' 127.0.0.1:5000/chatbot
# Another solution is to activate it as a webpage using GET as in the example
# change www.example.com endpoint to your chatbot route (http://127.0.0.1:5000/chatbot)in /static/script.js
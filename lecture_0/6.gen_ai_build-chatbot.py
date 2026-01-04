# Step 2: Import our required tools from the transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Step 3: Choosing a model
# Open-source license and runs relatively fast.
model_name = "facebook/blenderbot-400M-distill"
 # Step 4: Fetch the model and initialize a tokenizer
 # Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pretrained_vocab_files_map
# Step 5: Chat
# Step 5.1: Keeping track of conversation history
conversation_history = []
#Step 6: Repeat
while True:
    # Step 5.2: Encoding the conversation history
    history_string = "\n".join(conversation_history)
    # Step 5.3: Fetch prompt from user
    # Get the input data from the user
    input_text = input("> ")
    # Step 5.4: Tokenization of user prompt and chat history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    #print(f"inputs: \n {inputs} \n------")
    # Python dictionary is created  which contains special keywords that allow the model to properly reference its contents.
    # Step 5.5: Generate output from the model
    outputs = model.generate(**inputs)
    #print(f"outputs: \n {outputs} \n------")
    # Step 5.6: Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(f"response: \n {response} \n------")

    # Step 5.7: Update conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    #print(f"conversation_history: \n {conversation_history} \n ------")
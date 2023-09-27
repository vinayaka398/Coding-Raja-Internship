# Coding-Raja-Internship
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# Import the dataset
dataset = pd.read_csv('chatbot_dataset.csv')
# Preprocess the text data
dataset['question'] = dataset['question'].str.lower()
dataset['question'] = dataset['question'].str.replace('[^a-zA-Z0-9]', ' ')
dataset['question'] = dataset['question'].str.split()
dataset['question'] = dataset['question'].apply(lambda x: [word for word in x if word not in stopwords.words('english')])
dataset['question'] = dataset['question'].apply(lambda x: [PorterStemmer().stem(word) for word in x])
# Create the intent recognition model
intent_recognizer = nltk.NaiveBayesClassifier.train(dataset[['question', 'intent']])
# Create the response generation model
response_generator = nltk.FreqDist(dataset['response'])
# Create the chatbot
chatbot = Chatbot(intent_recognizer, response_generator)
# Interact with the user
while True:
    # Get the user's question
    question = input("You: ")
    # Preprocess the user's question
    question = question.lower()
    question = question.replace('[^a-zA-Z0-9]', ' ')
    question = question.split()
    question = [word for word in question if word not in stopwords.words('english')]
    question = [PorterStemmer().stem(word) for word in question]
    # Get the intent of the user's question
    intent = chatbot.intent_recognizer.classify(question)
    # Get the response to the user's question
    response = chatbot.response_generator.freq(intent)
    # Print the response to the user
    print("Chatbot:", response)

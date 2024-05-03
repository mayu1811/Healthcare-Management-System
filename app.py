import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.exceptions import InconsistentVersionWarning
import warnings
import pickle
import nltk
nltk.download('punkt')

warnings.simplefilter("error", InconsistentVersionWarning)

model = pickle.load(open('Disease_prediction.pkl','rb'))

cv = pickle.load(open('Count_vector.pkl','rb'))

label_encoder=pickle.load(open('Encoder.pkl','rb'))


def preprocess_text(text):
    tokens = word_tokenize(text)
    snowball_stemmer = SnowballStemmer('english')
    tokens = [snowball_stemmer.stem(token.lower()) for token in tokens if token.isalpha()]
    return ' '.join(tokens)
def find_DR(Symptoms,Time):
    # result = model.most_similar(input_data)
    # return result
    sample_text_processed = preprocess_text(Symptoms)

    # Transform the preprocessed sample text using the loaded vectorizer
    # cv = CountVectorizer(max_features=3000)
    sample_text_transformed = cv.transform([sample_text_processed])

    # Predict using the loaded model
    predicted_label_encoded = model.predict(sample_text_transformed)

    # Decode the predicted label

    predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

    x = predicted_label[0]
    y = Time
    dff = pd.read_csv(r'C:\Users\amanj\PycharmProjects\DuplicateQuestionPairs\DiseasePrediction\dummy_dataset.csv')
    # Filter the DataFrame to only include rows where 'Associated_Diseases' is 'x' and the doctor is available at time 'y'
    filtered_dff = dff[(dff['Associated_Diseases'] == x) & (dff['Arrival_Time'] <= y) & (dff['Departure_Time'] >= y)]

    # If no doctor is available at time 'y', print a message and exit
    if filtered_dff.empty:
        return ("No doctor is available at the specified time.")
    else:
        # Find the row with the maximum rating
        best_doctor = filtered_dff[filtered_dff['Rating'] == filtered_dff['Rating'].max()]

        # Get the 'Doctor_ID' and 'Contact_Number' of the best doctor
        best_doctor_id = best_doctor['Doctor_ID'].values[0]
        best_doctor_contact = best_doctor['Contact_Number'].values[0]

        return(
            f"The doctor with the highest rating who can treat disease \"{x}\" and is available at time \"{y}\" is Doctor {best_doctor_id}. You can contact them at {best_doctor_contact}.")


def main():
    st.title('Disease Prediction and Doctor Recommendation')

    Symptoms=st.text_input('Enter the Symptoms')
    Time = st.text_input('Enter time at which you want to see Doctor(Enter in HH:MM format)')
    result = []
    if st.button('Find'):
        st.write(find_DR(Symptoms,Time))  # Use st.text() instead of print()

    # st.success(result)

if __name__ == '__main__':  # Corrected from '_main_'
    main()

import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# Load pre-trained models
clf = pickle.load(open('my_classifier.pkl', 'rb'))
tfidfd = pickle.load(open('my_vectorizer.pkl', 'rb'))

# Preprocessing
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Main web app function
def main():
    # Create main container
    with st.container():
        # Create two columns
        col1, col2 = st.columns([1, 2])
        # Column 1: Display image
        
        col1.image('11.jpg', width=100)
        # Column 2: Display title
        col2.title('RESUME CLASSIFIER')

        uploaded_file = None
        resume_text = None

        # Upload resume file or enter resume text
        choice = st.sidebar.selectbox("CHOICE", ['Input File', 'Input Text'])

    
        if choice == 'Input File':
            uploaded_file = st.file_uploader('*Upload Resume*', type=['txt', 'pdf'])
        else:
            resume_text = st.text_area('*Paste Resume Text*', height=200)

        # Process and predict job category if either a file or text is provided
        if uploaded_file is not None or (resume_text is not None and resume_text.strip() != ""):
            if uploaded_file is not None:
                try:
                    resume_bytes = uploaded_file.read()
                    resume_text = resume_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    # If UTF-8 decoding fails, try decoding with 'latin-1'
                    resume_text = resume_bytes.decode('latin-1')
            resume_text = resume_text.lower()
            # Clean and preprocess resume text
            cleaned_resume = clean_resume(resume_text)

            # Convert text to TF-IDF features
            input_features = tfidfd.transform([cleaned_resume])

            # Predict job category using the trained classifier
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",23: "Testing",8: "DevOps Engineer",20: "Python Developer",
                24: "Web Designing",12: "HR",13: "Hadoop",3: "Blockchain",10: "ETL Developer",
                18: "Operations Manager",6: "Data Science",22: "Sales",16: "Mechanical Engineer",
                1: "Arts",7: "Database",11: "Electrical Engineering",14: "Health and Fitness",
                19: "PMO",4: "Business Analyst",9: "DotNet Developer",2: "Automation Testing",
                17: "Network Security Engineer",21: "SAP Developer",5: "Civil Engineer",0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")
            st.markdown("*Pridicted Category:* ")
            st.success( category_name)

# python main
if __name__ == "__main__":
    main()

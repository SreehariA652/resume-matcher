from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx

app = Flask(__name__)

# extracting text from uploaded files
def extract_text(file):
    text = ""
    if file.filename.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() #reads each page
    elif file.filename.endswith('.docx'):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"    #reads each paragraph
    else:
        text = file.read().decode('utf-8')
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match_resume():
    resume_file = request.files['resume']
    jd_text = request.form['job_description']
    jd_file = request.files.get('jd_file')

    if jd_file and jd_file.filename != '':
        jd_text = extract_text(jd_file)

    resume_text = extract_text(resume_file)

    # Calculating similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_text, resume_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

    # Finding missing keywords
    count_vec = CountVectorizer(stop_words='english')
    count_matrix = count_vec.fit_transform([jd_text, resume_text])
    words = count_vec.get_feature_names_out()
    jd_counts = count_matrix.toarray()[0]
    resume_counts = count_matrix.toarray()[1]
    missing_keywords = [words[i] for i in range(len(words)) if jd_counts[i] > 0 and resume_counts[i] == 0]

    return render_template('index.html', score=round(score, 2), missing=missing_keywords[:10])

if __name__ == '__main__':
    app.run(debug=True)
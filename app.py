from fastapi import FastAPI
from pydantic import BaseModel
from resume_parser import cleanResume, tfidf
import pickle

app = FastAPI()

clf = pickle.load(open("model/clf.pkl", "rb"))

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}


class ResumeText(BaseModel):
    text: str


@app.post("/predict")
def predict(data: ResumeText):
    cleaned = cleanResume(data.text)
    vector = tfidf.transform([cleaned])
    pred_id = clf.predict(vector)[0]
    return {"category": category_mapping.get(pred_id, "Unknown")}

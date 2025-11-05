**Movie Genre Classification — Leveraging AWS SageMaker & Bedrock**

## **Project Overview**
This project builds an **end-to-end machine learning pipeline** to classify movie genres based on their plot summaries or textual information.  
The model combines **classical ML techniques** such as **TF-IDF**, **word embeddings**, and classifiers like **Naive Bayes**, **Logistic Regression**, and **Support Vector Machines (SVM)** with modern **AWS cloud-based tools** for scalability, deployment, and explainability using **Amazon SageMaker** and **AWS Bedrock**.

The solution not only predicts genres but also **explains** why a movie was classified into a particular genre using **Generative AI (via Bedrock)** — bridging model interpretability and user engagement.

---

## **My Role**
I designed and implemented the complete ML workflow, from **data ingestion and preprocessing** to **model training**, **deployment**, and **explanation generation**.  
The implementation leverages **Amazon SageMaker** for model lifecycle management and **AWS Bedrock** for generating natural language insights on model predictions.

---

## **Architecture Overview**


Plot summaries → Amazon S3 (Raw Data Storage)
        ↓
SageMaker Studio / Data Wrangler → Feature Extraction (TF-IDF / Embeddings)
        ↓
SageMaker Training Jobs → Model Training & Tuning
        ↓
SageMaker Model Registry → Version Control & Tracking
        ↓
SageMaker Endpoint → Real-Time Inference API
        ↓
Bedrock (Foundation Model + Knowledge Base)
        ↓
User Interface / Chatbot → Genre Prediction + Natural Language Explanation
Implementation Details

1. Data Preparation & Feature Engineering
	•	Uploaded the dataset (movie plots + genre labels) to an S3 bucket.
	•	Used SageMaker Studio notebooks for:
	•	Text cleaning (tokenization, lowercasing, removing special characters).
	•	Feature extraction using TF-IDF and Word Embeddings (GloVe).
	•	Exported processed data back to S3 for reproducible experiments.

⸻

2. Model Training & Deployment with SageMaker
	•	Explored multiple classifiers:
	•	Naive Bayes, Logistic Regression, Support Vector Machines (SVM).
	•	Used SageMaker Training Jobs to automate training runs, store artifacts, and track model metrics.
	•	The best-performing model was registered in the SageMaker Model Registry.
	•	Deployed the model as a SageMaker Endpoint for real-time API inference.
from sagemaker import Session, Estimator, image_uris
from sagemaker.inputs import TrainingInput

sess = Session()
role = "<SAGEMAKER_EXECUTION_ROLE>"
bucket = "<YOUR_S3_BUCKET>"

xgb_image = image_uris.retrieve("xgboost", sess.boto_region_name, "latest")

estimator = Estimator(
    image_uri=xgb_image,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/artifacts/"
)

estimator.fit({"train": TrainingInput(f"s3://{bucket}/processed/train.csv", content_type="text/csv")})
3. Generative AI Layer with Bedrock
	•	Built a Knowledge Base containing:
	•	Movie genre definitions.
	•	Common plot-to-genre keyword mappings.
	•	Integrated Bedrock to generate explainable predictions using natural language.

Example Bedrock prompt:

“Given the movie plot and the predicted genre, explain in 2–3 sentences why the model classified it as that genre. Highlight key words or themes that influenced the classification.”

Bedrock then generates an output such as:

“The plot contains recurring themes of warfare, heroism, and revenge, which are strong indicators of the ‘Action’ genre. The model identified frequent use of combat-related verbs and tense pacing.”

⸻

4. API Orchestration & User Interface
	•	Created a lightweight API (AWS Lambda + API Gateway) that:
	1.	Accepts a movie plot.
	2.	Sends it to the SageMaker Endpoint for genre prediction.
	3.	Passes the plot and prediction to Bedrock.
	4.	Returns a user-friendly explanation with the predicted genre.

Example Response:
{
  "Genre": "Drama",
  "Explanation": "The model detected emotional narrative arcs and interpersonal conflict common in dramatic storytelling."
}
5. Monitoring & Production Readiness
	•	Enabled CloudWatch metrics for:
	•	Endpoint latency
	•	Invocation count
	•	Prediction drift monitoring
	•	Used SageMaker Model Monitor for detecting feature drift and retraining triggers.
	•	Controlled Bedrock token usage for cost management and prompt safety (avoiding hallucinations).

⸻

Evaluation & Metrics

The models were evaluated using:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-Score
Tech Stack
	•	Python (NumPy, Pandas, Scikit-learn, NLTK)
	•	AWS SageMaker — Model training, tuning, deployment
	•	AWS Bedrock — Generative AI layer
	•	Amazon S3 — Data storage
	•	AWS Lambda — API orchestration
	•	CloudWatch — Monitoring
	•	TensorFlow (optional for embeddings)

Setup & Installation
# Clone the repository
git clone https://github.com/afra16181falakh/Movie-genre-Classification.git
cd Movie-genre-Classification

# Install dependencies
pip install -r requirements.txt

# Run training
python train_model.py

Conclusion
This project demonstrates how traditional NLP-based classification can be scaled and enhanced through cloud-native AI services.
By combining SageMaker’s managed ML infrastructure with Bedrock’s generative reasoning, the pipeline delivers not only accurate genre predictions but also human-readable insights — bridging interpretability and production-grade deployment.

⸻

Future Enhancements
	•	Integrate multilabel genre prediction (movies can belong to multiple genres).
	•	Add semantic search for similar movie plots using Bedrock embeddings.
	•	Extend to movie recommendation system using genre similarity scores.

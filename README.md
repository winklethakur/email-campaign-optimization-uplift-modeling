# Email Campaign Optimization using Uplift Modeling & A/B Testing

This project is about figuring out which customers should actually receive a marketing email, instead of just blasting every customer and hoping someone responds. I used a real email marketing dataset called the Hillstrom MineThatData dataset, built an end to end analysis in a Jupyter notebook, trained an uplift model, and then deployed it as a REST API using FastAPI on Render.

The live API with Swagger UI is available here: https://email-uplift-predictor.onrender.com/docs

---

## Table of Contents

1. [Why I Built This](#why-i-built-this)
2. [Project Structure](#project-structure)
3. [The Dataset](#the-dataset)
4. [Notebook Walkthrough](#notebook-walkthrough)
5. [The API](#the-api)
6. [How to Run Locally](#how-to-run-locally)
7. [Deployment on Render](#deployment-on-render)
8. [Results](#results)
9. [Tech Stack](#tech-stack)
10. [What I Learned](#what-i-learned)

---

## Why I Built This

I wanted to work on something beyond the usual beginner projects like predicting house prices or classifying images. Experimentation and uplift modelling is something that real companies use to decide who to send emails to, who to show ads to, and which users to target with a new feature. I came across this topic and thought it would be a good challenge to actually implement from scratch.

The core idea behind uplift modelling is that not everyone responds to an email the same way. Some customers would have visited the website anyway. Some will never visit no matter what you send them. The only people worth targeting are the ones who will visit specifically because they received the email. Uplift modelling tries to find those people. I found that idea genuinely interesting so I built the whole thing out.

---

## Project Structure

```
uplift-modelling/
    api/
        main.py
        models.pkl
    notebook/
        hillstrom_uplift_analysis.ipynb
    requirements.txt
    render.yaml
    .gitignore
    README.md
```

The api folder contains the FastAPI application and the serialized model file. The notebook folder contains the full analysis notebook and the dataset. The files at the root level handle dependencies and deployment configuration.

---

## The Dataset

The dataset is from Kevin Hillstrom's MineThatData E-Mail Analytics Challenge. It is a real dataset from a real email experiment with 64,000 customers. Hillstrom shared it publicly for people to practice on.

The dataset is not included in this repository because of the file size. You can download it from Kaggle using the link below.

[Download the dataset from Kaggle](https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics?select=Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv)

The experiment randomly split customers into three groups. One group received a mens clothing email, one received a womens clothing email, and one received nothing. Two weeks later the outcomes were recorded.

The columns in the dataset are as follows.

**recency** is how many months ago the customer last made a purchase. It ranges from 1 to 12.

**history** is the total amount in dollars the customer has spent historically.

**mens** is a binary column indicating whether the customer has bought mens items before.

**womens** is the same but for womens items.

**zip_code** tells you whether the customer lives in a Rural, Suburban, or Urban area.

**newbie** is 1 if the customer was acquired in the past 12 months and 0 otherwise.

**channel** tells you how the customer was acquired, either Phone, Web, or Multichannel.

**history_segment** is a categorical version of the history column grouped into spend bands.

**segment** is the treatment assignment, either No E-Mail, Mens E-Mail, or Womens E-Mail.

**visit** is the primary outcome, a 1 if the customer visited the website within two weeks and 0 otherwise.

**conversion** is 1 if the customer made a purchase within two weeks.

**spend** is the amount spent within two weeks.

---

## Notebook Walkthrough

The notebook is in the notebook folder and is named hillstrom_uplift_analysis.ipynb. It runs from top to bottom as a complete pipeline. Here is what each section does.

### Section 1: Setup and Imports

This section just imports all the libraries needed throughout the notebook. It also sets up a consistent plot style so all the charts look clean.

### Section 2: Load and Explore the Data

Here I loaded the CSV file and did basic exploration. I looked at the data types, checked for missing values, and plotted distributions for the key features. I also compared visit rates and conversion rates across the three segments visually to get a first sense of whether the emails had an effect.

### Section 3: Feature Engineering and Preprocessing

This section prepares the data for modelling. The categorical columns like history_segment, zip_code, and channel need to be converted to numbers. I also created three new features. One is a binary flag for customers who have bought from both mens and womens categories, since these customers tend to respond more to email. Another is an interaction between recency and the log of history spend. The third is a binary flag for high value customers. The final feature set has 11 columns.

### Section 4: A/B Test Analysis

This is where I checked whether the email campaigns actually worked at a population level. I ran two sided t-tests comparing the visit rate, conversion rate, and average spend between each treatment group and the control group. I also calculated the absolute uplift, the relative uplift as a percentage, the p-value, and Cohen's d as a measure of practical effect size.

I also computed 95 percent confidence intervals around each uplift estimate. A p-value just tells you whether something is significant. The confidence interval tells you how big the effect actually is in the best and worst case, which is more useful for making decisions.

The mens email increased visit rate from 10.6 percent to 18.3 percent. The womens email increased it to 15.1 percent. Both were statistically significant with p values below 0.0001.

### Section 5: Power Analysis

Power analysis is something you are supposed to do before running an experiment to figure out how many users you need. I did it after the fact to understand whether the experiment was designed well.

The idea is that there are four things connected: the sample size, the minimum effect you want to be able to detect, the significance threshold, and the statistical power which is the probability of detecting a real effect when it exists. I used the two-proportion z-test formula to calculate how many users per group were actually needed.

It turned out the experiment needed only 331 users per arm to detect the mens email effect at 80 percent power. The actual experiment had over 21,000 per arm. That means the experiment was about 64 times more powerful than needed. The results are extremely reliable.

I also plotted the minimum detectable effect curve which shows how small an effect you could have detected at different sample sizes.

### Section 6: Uplift Modelling

This is the main section of the notebook. Instead of asking whether email works on average, I tried to estimate the effect for each individual customer.

I implemented two approaches. The first is the T-Learner which stands for Two-model Learner. The idea is to train a completely separate Gradient Boosting model on each experimental arm. The control model learns what predicts a visit when no email is sent. The treatment models learn what predicts a visit when an email is sent. The uplift for any customer is the prediction from the treatment model minus the prediction from the control model. A high uplift score means that customer is likely to visit because of the email, not for other reasons.

The second approach is the S-Learner which stands for Single-model Learner. Here I trained one model on all the data together, with the treatment assignment included as a feature. To get the uplift for a customer I predict twice, once pretending they are in the treatment group and once pretending they are in the control group, and take the difference.

Both approaches use Gradient Boosting as the base algorithm, with 100 trees, maximum depth of 3, and a learning rate of 0.1.

### Section 7: Model Evaluation

For the response models I evaluated AUC-ROC on a held-out test split. The AUC values came out around 0.64 for all three models. That is not very high but it is expected for behavioral data. Human decisions to visit a website have a lot of random noise in them.

For the uplift models I used the Qini curve which is the correct evaluation metric for uplift. The AUC-ROC cannot be used here because you never observe what would have happened to a treated customer without the treatment. The Qini curve works by ranking all customers by their predicted uplift score and then measuring how many incremental conversions you capture as you target more and more people. A model that ranks customers correctly will have a steep curve that rises quickly. A random model produces a diagonal line. The T-Learner performed better than the S-Learner on this dataset.

### Section 8: Feature Importance

This section plots the feature importances from the Gradient Boosting model trained on the mens email group. The most important features were whether the customer had bought mens items, whether they had bought womens items, and their spend history segment. The newbie feature had a negative effect, meaning new customers tend to respond less to email.

### Section 9: Uplift Segmentation

Here I divided all customers into 10 equal groups called deciles based on their predicted uplift score, from lowest to highest. Then I checked the actual observed visit rates in each decile for the treatment and control groups. The top decile showed about 13.3 percentage points of observed uplift, while the bottom decile showed only around 6.2 percentage points. This confirms that the model is correctly identifying which customers are more responsive.

I also profiled the top 20 percent and bottom 20 percent of customers by predicted uplift to understand what the most responsive customers look like. High uplift customers tend to have bought from both categories, purchased more recently, and spent more historically.

### Section 10: Decision Framework

After all the analysis I made an explicit decision about what to actually do. The recommendation is to send the mens email to all customers since every segment shows a positive effect. For the womens email the recommendation is to target only the top 20 percent by uplift score since the average effect is weaker. Running the experiment longer is not necessary because we already have far more data than needed.

### Section 11: Score a New Customer

This section has a function called score_customer that takes a customer's features as input and prints out the predicted visit probabilities under each campaign and a recommendation. It is a quick way to test the model interactively inside the notebook.

### Section 12: Save Models to Pickle

This final section serializes everything the API needs into a single file called models.pkl using joblib. It saves the fitted scaler, the three trained models, the population baseline rates, the feature list, and the encoding maps for the categorical columns. It then does a round-trip verification by loading the file back and running an inference to confirm everything works correctly.

---

## The API

The API is built with FastAPI and is in the api/main.py file. It loads models.pkl at startup and keeps it in memory for fast inference.

### Endpoints

**GET /health** returns a simple status check to confirm the API is running.

**GET /baselines** returns the population level visit and conversion rates from the original experiment across all three arms.

**POST /predict** takes a single customer's details and returns the predicted visit probability under each campaign along with a recommendation.

**POST /predict/batch** accepts a list of up to 10,000 customers and returns predictions for all of them. It also returns summary counts of how many customers fall into each recommendation category.

### Input Fields for the Predict Endpoint

| Field | Type | Allowed Values |
|-------|------|----------------|
| recency | integer | 1 to 12 |
| history | float | any positive number |
| mens | integer | 0 or 1 |
| womens | integer | 0 or 1 |
| zip_code | string | Rural, Surburban, Urban |
| newbie | integer | 0 or 1 |
| channel | string | Phone, Web, Multichannel |
| history_segment | string | one of these seven values: "1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500", "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +" |

### Recommendation Logic

The API returns one of three recommendations based on the best predicted uplift across both campaigns.

Send means the best uplift is 8 percentage points or higher. This customer is likely a persuadable who will respond to the email.

Borderline means the best uplift is between 3 and 8 percentage points. It is worth considering whether the email cost is justified.

Skip means the best uplift is below 3 percentage points. Budget is better spent on other customers.

### Example Request and Response

Request body for POST /predict:

```json
{
  "recency": 3,
  "history": 650.0,
  "mens": 1,
  "womens": 1,
  "zip_code": "Rural",
  "newbie": 0,
  "channel": "Multichannel",
  "history_segment": "5) $500 - $750"
}
```

Response:

```json
{
  "p_visit_no_email": 0.142,
  "p_visit_mens_email": 0.231,
  "p_visit_womens_email": 0.198,
  "uplift_mens": 0.089,
  "uplift_womens": 0.056,
  "best_campaign": "Mens email",
  "best_uplift": 0.089,
  "recommendation": "Send"
}
```

---

## How to Run Locally

Clone the repository.

```
git clone https://github.com/yourusername/uplift-modelling.git
cd uplift-modelling
```

Install the dependencies.

```
pip install -r requirements.txt
```

Move into the api folder and start the server.

```
cd api
uvicorn main:app --reload
```

Open your browser and go to http://localhost:8000/docs to use the Swagger UI.

---

## Deployment on Render

I pushed the code to a GitHub repository and connected it to Render. Render reads the render.yaml file at the root of the project and uses it to set the build and start commands automatically.

The build command is:

```
pip install -r requirements.txt
```

The start command is:

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The app is running on Render's free tier. Free tier instances spin down after 15 minutes of inactivity, so the first request after that will take around 30 seconds to respond while the server wakes up. After that it responds instantly.

---

## Results

The mens email campaign increased the website visit rate from 10.6 percent to 18.3 percent, which is a 72 percent relative improvement. The conversion rate went from 0.57 percent to 1.25 percent, which is a 119 percent improvement. Both results were statistically significant with p values well below 0.0001.

The womens email increased visit rate to 15.1 percent, a 43 percent improvement, and conversion rate to 0.88 percent.

The uplift model showed that the top 20 percent of customers by predicted uplift respond about twice as strongly as the bottom 20 percent. High uplift customers tend to have bought from both categories, purchased recently, and spent more historically.

The overall recommendation from the analysis was to send the mens email to everyone since every segment shows a positive effect, and to restrict the womens email to the top 20 percent of customers by predicted uplift.

---

## Tech Stack

FastAPI was used for the API because it is fast, gives you automatic Swagger documentation, and has good support for type validation through Pydantic.

Scikit-learn was used for the Gradient Boosting models. I used GradientBoostingClassifier with 100 estimators, max depth of 3, and learning rate of 0.1.

Joblib was used to save and load the trained models.

NumPy was used for the feature calculations.

Uvicorn is the ASGI server that runs the FastAPI application.

Render is the platform where the API is deployed.

---

## What I Learned

This project taught me a lot that I had not encountered before. A/B test analysis was something I had heard of but never implemented properly. The difference between statistical significance and practical significance was not clear to me before this, and I now understand why p values alone are not enough to make a decision.

Power analysis was completely new to me. I did not know that you are supposed to calculate sample size requirements before running an experiment. Learning that the Hillstrom experiment was far more powered than necessary was an interesting finding.

Uplift modelling was the most challenging part. Understanding why you cannot just train one model and why you need separate models per arm, or alternatively include the treatment as a feature and predict counterfactually, took some time to get right.

Deploying the model with FastAPI was also new for me. Getting joblib serialization to work correctly so the scaler and models load properly inside the API, and making sure the feature engineering steps in the API matched exactly what was done during training, required careful attention.

---

## Author

[Winkle]

[LinkedIn](https://www.linkedin.com/in/winkle-data-scientist/)

[GitHub](https://github.com/winklethakur)
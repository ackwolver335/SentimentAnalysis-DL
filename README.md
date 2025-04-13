# 📰 Sentiment Analysis - Deep Learning 🗯️

#### Sentence Based Sentiment Analysis

📌 A sentence-level sentiment analysis system built in **Python** using a custom deep learning model inspired by **ResNet** and **GoogleNet** architectures. This project performs sentence-level sentiment analysis using a custom deep learning model built in Python.

📌 The architecture takes inspiration from ResNet and GoogleNet, enabling effective feature extraction from text for classifying sentiments as Positive, Negative, or Neutral.

## 📤 Project Description

📄 This project is focused on sentence-based sentiment analysis using a deep learning approach developed entirely in Python. Instead of relying on traditional machine learning models, it introduces a custom neural network architecture inspired by **ResNet** and **GoogleNet**, two of the most powerful models in computer vision — adapted here for **Natural Language Processing (NLP)** tasks.

📃 The model is designed to analyze the sentiment of individual sentences and classify them into one of three categories: **Positive, Negative,** or **Neutral**. Through a combination of convolutional and residual learning concepts, the model efficiently captures semantic patterns and contextual meanings in the input text.

#### 🖥️ Real World Applications ⚙️

➡️ **Customer Feedback Analysis** → Used in **Business Analysis** Reviews in order to improve the products & services, as per customer's feedback.

➡️ **Social Media Monitoring** → In order to analyse **Public's Sentiments** regarding their feeds and **Brands Gauging** over to the Platforms like **Twitter**, **Facebook**, **Instagram**,..etc from their reviews or feedback (Also from the Comments).

➡️ **HealthCare** → Used for Identification of **Patient's Emotions** in feedback in order to enhance care and treatments quality.

➡️ **Market Research** → In order to understand the **Customer's** or **Consumer's** preferences through the survey or feedback responses.

➡️ **Chatbots** → Used for Detection of user's responses while providing responses or requirements in a better or enhanced way.

## 🔖 Demo / Example Output 📮

📄 Providing you the demo example regarding an overview of **" How this Model works ? "**, example is shown below

📍Remember to go in this **[resnet_model](https://github.com/ackwolver335/SentimentAnalysis-DL/tree/main/resnet_model)** folder before running the code below for getting sample result ready 👍🏻

```bash
python model_test1.py
```

📤 **Code Output** →

```bash
Text: I'm extremely unhappy with how things turned out.
Predicted Sentiment: NEGATIVE
Confidence Score: 0.99
```

#### Sample Output Image

![Image](https://github.com/user-attachments/assets/391656f9-ae89-4a71-a3bb-49dc82dd87b1)

📝 **Remember** : This particular sentiment is analysed using one model only, but as both the model's are good, we could gather a good accuracy with the textual content that we have provided.

## 📑Features of this Model 💻

#### List of Main Capabilities

▶️ **Deep Learning Based Sentiment Detection** → Uses advance **Neural Network** for classification of sentiments detection.

▶️ **Preprocessing Pipeline** → In the Model's creation code, recleaning of data, together by Generating the **DL Model**.

▶️ **Supports Inference & Training Mode** → Contains both **Sentiment's Detection** & Model's **Retraining**.

▶️ **Multi-Class Classification** → Differentiates b/w **Positive**, **Negative** & **Neutral** Sentiments effectively.

## 💻 Tech Stack 📚

📌 Programming Language → **Python 3.12**

📌 Libraries/Packages → **Numpy**, **Pandas**, **matplotlib**, **seaborn**, **sklearn** & **tensorflow**.

## 📦 Project's Structure 💻

→ Initially, we have **3 Branches** in our main Repository from which each branch have its own workflow. Below we have the detailed description of these branches together by their work usages.

| 📝 **Branches** | 📮**Usage or Workflow** | 🖥️ **Status** |
| :-------------- | :---------------------- | -------------- |
| **main** | 🔖 Development of DL Model using **Resnet** | ✅ **Completed** |
| **combined** | 🔖 Development of DL Model using **GoogleNet** | ✅ **Completed & Merged** |
| **docs** | 🔖 Documentation workflow regarding **Synopsis** & 📄 **Research Paper** | **In Progress** |

## ⚙️ Installation & Setup ▶️

#### Below 👇🏻 we have Guide regarding Project's Setup in Steps

▶ **Cloning the Repository** → It is the first & easiest step regarding the usage of the Project in your Local Device for which the command in mentioned below.

```bash
git clone repository_https_link
```

▶ **Environment Setup** → Remember to install all the required **Modules** or **Libraries** in your Local Device with the **Python IDLE** or you can also install the virtual environment not to get the data meshed all over the Device.

📳 Installing Module default in **Local Device** 👇🏻

```bash
pip install library_name or module_name
```

📳 Creating a **Virtual Environment** & installing **Modules** or **Libraries** 👇🏻

```bash
pip install virtualenv
```

📌 Then you need to create & activate the **Virtual Environment** before installing the **Modules** or **Libraries** 👇🏻

```bash
python -m venv myenv_name                   # creating the virtual environment folder
myenv_name\Scripts\activate                 # activating the environment

pip install module_name or library_name     # installation will only be at the Virtual environment
```

▶ **Installing Dependencies** → Regarding the required **Modules** or **Libraries** that are mentioned below with the above commands, you just need to install those either in your **Local Device's System or User's variables** or in a **Virtual Environment** setup as shown above.

▶ **Running the Code** → You don't have to run the overall code, if you are properly cloning this whole **Repository**. You just need to run this **[code](https://github.com/ackwolver335/SentimentAnalysis-DL/blob/main/resnet_model/model_test1.py)** regarding the **Resnet** Developed Model. But, if you want to go with the Model developed using **GoogleNet** you can go with this **[code](https://github.com/ackwolver335/SentimentAnalysis-DL/blob/main/googleNet_model/model_test1.py)**.

## ▶ Usage Instruction 📮

#### ➡️ How to train the Model (if necessary) ?

🔖 In order to train the Model, you need to have all the **Dependencies [Libraries & Models]** already installed or if not the instructions are mentioned above [check](#️-installation--setup-️). As it is not necessary for training & recreating as the model is already created, but if you wanted to, in case of **Resnet** you need to first run this [code](https://github.com/ackwolver335/SentimentAnalysis-DL/blob/main/resnet_modelGn/trial_one.py) for generating the model & then [code](https://github.com/ackwolver335/SentimentAnalysis-DL/blob/main/resnet_model/model_loading.py) for generating the Tokenizer.

🔖 Now you have both the model & the tokenizer, put them in the same folder & then run this [code](https://github.com/ackwolver335/SentimentAnalysis-DL/blob/main/resnet_model/model_test1.py), it contains a default sentence, as if you wanted some you can change it on your choice.

#### 💻 Command Usage with Worflow 📰

📍 As most of the usable and script related commands are mentioned above only, then also for running the code, below we have the commands. And remember to go on proper locations in the folders before running the scripts.

```bash
# In resnet_model's folder
python model_test1.py

# In googleNet_model's folder
python model_test1.py
```

## 📕 Model Performance ⚙️

➡️ Our Model regarding Sentiment Analysis have few things to consider in order to get Comparison happen in b/w both the **DL Models** and their **Architecture** used here regarding the **Resnet** & **GoogleNet**.

| 📰 **Models** | **Accuracy** |
| :------------ | :----------- |
| 1️⃣ **Resnet** | 💻 **84%** |
| 2️⃣ **GoogleNet** | 💻 **81.8%** or **82%** |

🔖 Regarding the Information of the Confusion Metrics for both the Models the images are shown below 👇🏻

#### Resnet Based Model's Generated Confusion Matrix

![Image](https://github.com/user-attachments/assets/8c996938-da0f-4c00-9b26-19198f90528e)

📝 **Note** : Other visuals regarding **Confusion Matrix** & **Classification Report** for **Resnet** is in this [folder](https://github.com/ackwolver335/SentimentAnalysis-DL/tree/main/resnet_visualWF) and for **GoogleNet** in this [folder](https://github.com/ackwolver335/SentimentAnalysis-DL/tree/main/googleNet_visuals).

## 🖥️ Dataset 📄

📌 Our Dataset is that initial one, which was used in the last **Machine Learning** Project for the same **Sentiment Analysis** together with which it is available in both the folder of **GoogleNet** or **Resnet** based architecture, go through this [link](https://github.com/ackwolver335/SentimentAnalysis-DL/tree/main/resnet_modelGn) as it is a csv file with name **processed_data.csv**.

## 📦 Testing & Evaluation 📰

✒️ Regarding the Testing & Evaluation as per the purpose of checking if the generated model is working correctly or not, let's take an example here with a simple test case : **I'm extremely unhappy with how things turned out.** and for each model let's test it one by one.

#### 🔖 Using this Test Case with Resnet based Deep Learning Model

✏️ Remember to be at this particular [location](https://github.com/ackwolver335/SentimentAnalysis-DL/tree/main/resnet_model) before running the Script or command given below 👇🏻

```bash
# command to follow
python model_test1.py

# expected output
Text: I'm extremely unhappy with how things turned out.
Predicted Sentiment: NEGATIVE
Confidence Score: 0.99
```

#### 🔖 With GoogleNet based Architecture

✏️ Follow the commands given below 👇🏻

```bash
# command for running model generation code
python model_test1.py

# output regarding it
Text: I'm extremely unhappy with how things turned out.
Predicted Sentiment: NEGATIVE
Confidence Score: 0.70
```

## 👨‍💻 Limitations & Todo 🖥️

#### Current Limitations regarding this generated Model

➡️ The Architecture used here is not properly suitable for **Text Classification**, also requries significant adaptation.

➡️ As **Convolutional Layers** in **Resnet** or **GoogleNet** are well optimized for spatial features, not for sequential language patterns.

➡️ Without proper **pre-processing**, accuracy may suffer compared to **NLP-Specific** Modals.

➡️ **Computational Overhead** may increase due to handling text in a framework meant for **Image Data**.

#### 📑 Improvements regarding Future Enhancement ✒️

🖋️ Adaptation of **Resnet** & **GoogleNet** regarding text embedding instead of **Raw Words**.

🖋️ To explore Hybrid approaches in order to integrate **CNN Layers** with **LSTMs** or **Transformers**.

🖋️ **Proper Architectural** adjustments similar to increasing the kernel sizes for sequential data.

🖋️ Benchmark Performance against **NLP specific Models** in order to justify their usage regarding **Sentiment Analysis**.

## 👨‍💻 Authors or Contributors 🖥️

| 📄 **Names** | **Work Fields** | 📑 **Github IDs** |
| :----------- | :-------------- | :---------------- |
| **Abhay Chaudhary** | **Python Developer (AI/ML)** | **[ackwolver335](https://github.com/ackwolver335)** |
| **Abhishek Chautala** | **Python Developer (AL/ML)** | **[sirChautala](https://github.com/sirCHAUTALA)** |
| **Sankalp Sharma** | **Python Developer (Data Science & Analytics)** | **[sankalp1046](https://github.com/sankalp1046)** |
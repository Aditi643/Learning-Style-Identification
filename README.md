This a model made for predicting the learning style of an individual by answering few questionnaire and uses different algorithm like knn, svm ,random forest etc and also uses hybrid algorithms by combining svm,nn etc There are 3 major folders frontend -html,css,js backend -requirments, main.py using fast api Learning style prediction models

Steps to install in your system (windows)

Download as ZIP:
Go to the GitHub repository.
Click "Code" → "Download ZIP".
Extract it to a desired folder.
or

in command prompt first go to your folder in which you want to clone for changing to d drive type D: now cd your folder name

now you are in the folder

activate virtual environment python -m venv venv venv\Scripts\activate

now install libraries pip install flask streamlit pandas scikit-learn numpy joblib imbalanced-learn matplotlib

8.(any other if propmted then install it later) 9. to run in terminal use python modelname.py

to start backend cd .. (to move one folder back)

go to backend_api folder and type uvicorn main:app --host 0.0.0.0 --port 8000 --reload this runs api locally at this port http://127.0.0.1:8000

9.so start the backend from command prompt and then run the live server of html from vs code open frontend folder in vs code

answer the questions and click on predict and result is displayed
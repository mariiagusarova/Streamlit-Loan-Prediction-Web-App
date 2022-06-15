import streamlit as st
import pandas as pd
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle as pkl
import shap
import streamlit.components.v1 as components

#Load the saved model
model=pkl.load(open("model.p","rb"))

st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="/Users/mariia/Desktop/AWS/loan.png"
)

st.set_option('deprecation.showPyplotGlobalUse', False)

######################
#main page layout
######################

st.title("Loan Default Prediction")
st.subheader("Are you sure your loan applicant is surely going to pay the loan back?💸 "
                 "This machine learning app will help you to make a prediction to help you with your decision!")

col1, col2 = st.columns([1, 1])

with col1:
    st.image("/Users/mariia/Desktop/AWS/loan.png")

with col2:
    st.write("""To borrow money, credit analysis is performed. Credit analysis involves the measure to investigate
the probability of the applicant to pay back the loan on time and predict its default/ failure to pay back.

These challenges get more complicated as the count of applications increases that are reviewed by loan officers.
Human approval requires extensive hour effort to review each application, however, the company will always seek
cost optimization and improve human productivity. This sometimes causes human error and bias, as it’s not practical
to digest a large number of applicants considering all the factors involved.""")

st.subheader("To predict default/ failure to pay back status, you need to follow the steps below:")
st.markdown("""
1. Enter/choose the parameters that best descibe your applicant on the left side bar;
2. Press the "Predict" button and wait for the result.

""")

st.subheader("Below you could find prediction result: ")

######################
#sidebar layout
######################

st.sidebar.title("Loan Applicant Info")
st.sidebar.image("/Users/mariia/Desktop/AWS/ab.png", width=100)
st.sidebar.write("Please choose parameters that descibe the applicant")

#input features
term = st.sidebar.radio("Select Loan term: ", ('36months', '60months'))
loan_amnt =st.sidebar.slider("Please choose Loan amount you would like to apply:",min_value=1000, max_value=40000,step=500)
emp_length = st.sidebar.selectbox('Please choose your employment length', ("< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                                                                           "6 years", "7 years","8 years","9 years","10+ years") )
annual_inc =st.sidebar.slider("Please choose your annual income:", min_value=10000, max_value=200000,step=1000)
sub_grade =st.sidebar.selectbox('Please choose grade', ("A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3","B4","B5","C1", "C2", "C3",
                                                        "C4", "C5", "D1", "D2", "D3","D4","D5", "E1", "E2", "E3","E4","E5","F1", "F2",
                                                        "F3", "F4", "F5", "G1", "G2", "G3","G4","G5"))
dti=st.sidebar.slider("Please choose DTI:",min_value=0.1, max_value=100.1,step=0.1)
mths_since_recent_inq=st.sidebar.slider("Please choose your mths_since_recent_inq:",min_value=1, max_value=25,step=1)
revol_util=st.sidebar.slider("Please choose revol_util:",min_value=0.1, max_value=150.1,step=0.1)
num_op_rev_tl=st.sidebar.slider("Please choose num_op_rev_tl:",min_value=1, max_value=50,step=1)

def preprocess(loan_amnt, term, sub_grade, emp_length, annual_inc, dti, mths_since_recent_inq, revol_util, num_op_rev_tl):
    # Pre-processing user input

    user_input_dict={'loan_amnt':[loan_amnt], 'term':[term], 'sub_grade':[sub_grade], 'emp_length':[emp_length], 'annual_inc':[annual_inc], 'dti':[dti],
                'mths_since_recent_inq':[mths_since_recent_inq], 'revol_util':[revol_util], 'num_op_rev_tl':[num_op_rev_tl]}
    user_input=pd.DataFrame(data=user_input_dict)


    #user_input=np.array(user_input)
    #user_input=user_input.reshape(1,-1)

    cleaner_type = {"term": {"36months": 1.0, "60months": 2.0},
    "sub_grade": {"A1": 1.0, "A2": 2.0, "A3": 3.0, "A4": 4.0, "A5": 5.0,
    "B1": 11.0, "B2": 12.0, "B3": 13.0, "B4": 14.0, "B5": 15.0,
    "C1": 21.0, "C2": 22.0, "C3": 23.0, "C4": 24.0, "C5": 25.0,
    "D1": 31.0, "D2": 32.0, "D3": 33.0, "D4": 34.0, "D5": 35.0,
    "E1": 41.0, "E2": 42.0, "E3": 43.0, "E4": 44.0, "E5": 45.0,
    "F1": 51.0, "F2": 52.0, "F3": 53.0, "F4": 54.0, "F5": 55.0,
    "G1": 61.0, "G2": 62.0, "G3": 63.0, "G4": 64.0, "G5": 65.0,
    },
    "emp_length": {"< 1 year": 0.0, '1 year': 1.0, '2 years': 2.0, '3 years': 3.0, '4 years': 4.0,
    '5 years': 5.0, '6 years': 6.0, '7 years': 7.0, '8 years': 8.0, '9 years': 9.0,
    '10+ years': 10.0 }
    }

    user_input = user_input.replace(cleaner_type)

    return user_input

#user_input=preprocess
user_input=preprocess(loan_amnt, term, sub_grade, emp_length, annual_inc, dti, mths_since_recent_inq, revol_util, num_op_rev_tl)

#predict button

btn_predict = st.sidebar.button("Predict")

if btn_predict:
    pred = model.predict_proba(user_input)[:, 1]

    if pred[0] < 0.78:
        st.error('Warning! The applicant has a high risk to not pay the loan back!')
    else:
        st.success('It is green! The aplicant has a high probability to pay the loan back!')

    #prepare test set for shap explainability
    loans = st.cache(pd.read_csv)("mycsvfile.csv")
    X = loans.drop(columns=['loan_status','home_ownership__ANY','home_ownership__MORTGAGE','home_ownership__NONE','home_ownership__OTHER','home_ownership__OWN',
                   'home_ownership__RENT','addr_state__AK','addr_state__AL','addr_state__AR','addr_state__AZ','addr_state__CA','addr_state__CO','addr_state__CT',
                   'addr_state__DC','addr_state__DE','addr_state__FL','addr_state__GA','addr_state__HI','addr_state__ID','addr_state__IL','addr_state__IN',
                   'addr_state__KS','addr_state__KY','addr_state__LA','addr_state__MA','addr_state__MD','addr_state__ME','addr_state__MI','addr_state__MN',
                   'addr_state__MO','addr_state__MS','addr_state__MT','addr_state__NC','addr_state__ND','addr_state__NE','addr_state__NH','addr_state__NJ',
                   'addr_state__NM','addr_state__NV','addr_state__NY','addr_state__OH','addr_state__OK','addr_state__OR','addr_state__PA','addr_state__RI',
                   'addr_state__SC','addr_state__SD','addr_state__TN','addr_state__TX','addr_state__UT','addr_state__VA','addr_state__VT', 'addr_state__WA',
                   'addr_state__WI','addr_state__WV','addr_state__WY'])
    y = loans[['loan_status']]
    y_ravel = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y_ravel, test_size=0.25, random_state=42, stratify=y)

    st.subheader('Result Interpretability - Applicant Level')
    shap.initjs()
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(user_input)
    fig = shap.plots.bar(shap_values[0])
    st.pyplot(fig)

    st.subheader('Model Interpretability - Overall')
    shap_values_ttl = explainer(X_test)
    fig_ttl = shap.plots.beeswarm(shap_values_ttl)
    st.pyplot(fig_ttl)
    st.write(""" In this chart blue and red mean the feature value, e.g. annual income blue is a smaller value e.g. 40K USD,
    and red is a higher value e.g. 100K USD. The width of the bars represents the number of observations on a certain feature value,
    for example with the annual_inc feature we can see that most of the applicants are within the lower-income or blue area. And on axis x negative SHAP
    values represent applicants that are likely to churn and the positive values on the right side represent applicants that are likely to pay the loan back.
    What we are learning from this chart is that features such as annual_inc and sub_grade are the most impactful features driving the outcome prediction.
    The higher the salary is, or the lower the subgrade is, the more likely the applicant to pay the loan back and vice versa, which makes total sense in our case.
    """)

st.write("""

**Author:Mariia Gusarova**

You could find more about this project on Medium [here].
""")

import streamlit
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

streamlit.set_page_config(page_title="Cosmic Object Classifier")

streamlit.markdown(
    """
    <div style="padding: 10px; background: linear-gradient(to right, #0077b6, #00b4d8); border-radius: 10px; color: white;">
        <h1 style="margin: 0; padding: 0.5rem; text-align: center;">🔭 Cosmic Object Classifier</h1>
    </div>
    """,
    unsafe_allow_html=True
)

data = pandas.read_csv("C:/Users/ASUS/Desktop/Project/star_classification.csv")
df = pandas.DataFrame(data)


df = df.drop(columns=["obj_ID","run_ID","rerun_ID","cam_col","field_ID","spec_obj_ID","plate","MJD","fiber_ID"],axis = 1)


le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])

X = df.drop("class", axis=1)
Y = df["class"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier()
model = knn.fit(X_train,Y_train)
y_pred = model.predict(X_test)

print(accuracy_score(Y_test,y_pred))

def input_features():
    with streamlit.container(border=True):
        streamlit.subheader("Enter Spectroscopic Features")
        
        col1, col2 = streamlit.columns(2)
        
        with col1:
            alpha = streamlit.number_input("Alpha (Right Ascension)", min_value=-180.0, max_value=180.0, value=130.0, step=0.01, format="%.4f")
            delta = streamlit.number_input("Delta (Declination)", min_value=-90.0, max_value=90.0, value=7.0, step=0.01, format="%.4f")
            u = streamlit.number_input("u (Ultraviolet Band)", min_value=-10.0, value=19.0, step=0.01, format="%.4f")
            g = streamlit.number_input("g (Green Band)", min_value=-10.0, value=18.0, step=0.01, format="%.4f")
            
        with col2:
            r = streamlit.number_input("r (Red Band)", min_value=-10.0, value=17.0, step=0.01, format="%.4f")
            i = streamlit.number_input("i (Near-Infrared Band)", min_value=-10.0, value=16.5, step=0.01, format="%.4f")
            z = streamlit.number_input("z (Infrared Band)", min_value=-10.0, value=16.0, step=0.01, format="%.4f")
            redshift = streamlit.number_input("Redshift", min_value=-5.0, value=0.1, step=0.001, format="%.4f")

    data = {
        "alpha" : alpha,
        "delta" : delta,
        "u" : u,
        "g" : g,
        "r" : r,
        "i" : i,
        "z" : z,
        "redshift" : redshift
    }

    features = pandas.DataFrame(data,index = [0])  
    
    return features


input_df = input_features()

streamlit.markdown("---")
main_col, pred_col = streamlit.columns([3, 1])

with main_col:
    streamlit.info("Click the button below to classify the celestial object based on the entered photometric data.")
    
    if streamlit.button("Classify Object", type="primary") :
        
        try:
            input_scaled = scaler.transform(input_df)
            result = knn.predict(input_scaled)
            
            if result[0] == 0:
                final_result = "Star"
                color = "orange"
                emoji = "⭐"
                help_text = "This object is classified as a Star."
            elif result[0] == 1:
                final_result = "Galaxy"
                color = "blue"
                emoji = "🌌"
                help_text = "This object is classified as a Galaxy."
            elif result[0] == 2:
                final_result = "QSO (Quasar)"
                color = "#8b0000"
                emoji = "💫"
                help_text = "This object is classified as a Quasi-Stellar Object (Quasar)."
            else:
                final_result = "Unknown"
                color = "gray"
                emoji = "❓"
                help_text = "Classification result is unexpected."

            # 4. Display result using st.metric for visual impact
            with pred_col:
                 streamlit.markdown("### Result")
                 streamlit.metric(
                     label="Predicted Class", 
                     value=final_result,
                     delta=emoji
                 )
                 streamlit.markdown(f'<p style="color:{color}; font-weight:bold;">{help_text}</p>', unsafe_allow_html=True)
                 
            
        except Exception as e:
            streamlit.error(f"An error occurred during prediction: {e}. Please check your inputs.")

with streamlit.expander("Show Raw Input Data"):
    streamlit.dataframe(input_df)


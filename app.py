from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            age=int(request.form.get('age')),
            fnlwgt=int(request.form.get('fnlwgt')),
            capital_gain=int(request.form.get('capital_gain')),
            capital_loss=int(request.form.get('capital_loss')),
            hours_per_week=int(request.form.get('hours_per_week')),
            workclass = request.form.get('workclass'),
            education= request.form.get('education'),
            education_num = request.form.get('education_num'),
            marital_status = request.form.get('marital_status'),
            occupation = request.form.get('occupation'),
            relationship = request.form.get('relationship'),
            race = request.form.get('race'),
            sex = request.form.get('sex'),
            native_country = request.form.get('native_country')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
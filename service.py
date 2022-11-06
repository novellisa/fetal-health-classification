import bentoml

from bentoml.io import JSON 

model_ref = bentoml.sklearn.get("fethal_health_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("fethal_health_classifier", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def classify(application_data):   # assumes application_data are already scaled
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)
    
    result = prediction[0]
    
    if result == 1:
    
        return {"Fethal health": "Normal"}
        
    elif result == 2:
    
        return {"Fethal health": "Suspect"}
        
    elif result == 3:
    
        return {"Fethal health": "Pathological"}
    


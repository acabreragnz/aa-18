from unittest import TestCase
import pandas as pd
from io import StringIO
import math
from lab4.ej3.src.anomaly_detector import AnomalyTextDetector


class TestAnomalyDectectorSimpleExample(TestCase):
    
    def test_anomaly_detector(self):
        path = './../datasets/basic_example.txt'

        with open(path, encoding='latin-1') as f:
            text = f.read()

        df = pd.read_csv(StringIO(text), header=None, sep='|')
        training_examples = df.iloc[:, 2]
        training_examples = training_examples.replace(r'http\S+', '', regex=True)

        instance = "drugs ebola and diabetes"

        atd = AnomalyTextDetector(threshold=2.0)
        atd.fit(training_examples)
        prediction = atd.predict(instance)
        prediction_is_anomalous = prediction[0]
        prediction_pdf = prediction[1]

        numpy_pdf = atd.numpy_productorial_pdf(instance)

        print(f'Prediction is anomalous: {prediction_is_anomalous}')
        print(f'Prediction pdf: {prediction_pdf}')
        print(f'Numpy pdf: {numpy_pdf}')

        # este metodo no es bueno para detectar anomalias en textos
        assert not prediction_is_anomalous
        assert math.isclose(prediction_pdf, numpy_pdf, rel_tol=1e-7)


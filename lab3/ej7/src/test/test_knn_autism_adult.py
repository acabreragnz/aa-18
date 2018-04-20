from unittest import TestCase
from lab3.ej7.src.classifier import KNNClassifier
from arff_helper import DataSet
from ds_preprocessing import DataSetPreprocessor


class TestKnnAutismAdult(TestCase):
    
    def test_basic(self):
        """
        En el conjunto de entrenamiento la clasificacion debe ser perfecta (para una instancias x,
        la instancia  que mas se acerca es si misma)
        """
        ds: DataSet = DataSet()
        ds.load_from_arff('./../../datasets/Autism-Adult-Training-Subset.arff')
        target_attribute = 'Class/ASD'
        preprocessor = DataSetPreprocessor(ds, target_attribute)
        df = preprocessor.transform_to_rn()

        classifier = KNNClassifier(1, target_attribute)
        classifier.fit(df)

        for i in range(len(df)):
            self.assertEqual(classifier.predict(df.loc[i]),
                             df.loc[i][target_attribute],
                             f'El valor predecido para la instancia {i} no coincide su valor en el dataset')


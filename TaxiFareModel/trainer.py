from cmath import log
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LassoCV , LogisticRegression
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib

class Trainer():
    def __init__(self, X, y,model):
        """
            X: pandas DataFrame
            y: pandas Series
        """

        self.experiment_name  = "[Gm] [munich] [Morad4444] LinearRegression + 783"
        self.pipeline = None
        self.X = X
        self.y = y
        self.model = model

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
         ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
         ])
        preproc_pipe = ColumnTransformer([
             ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
             ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', self.model)
             ])
        return pipeline

    def run(self):
        """set and train the pipeline"""
        pipeline = self.set_pipeline()
        pipeline.fit(self.X, self.y)
        return pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        pipeline = self.run()
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse
    def evaluateClassification(self, X_test, y_test):
        pipeline = self.run()
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        pipeline = self.run()
        joblib.dump(pipeline, 'model.joblib')
        print('model_saved')
    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":


    lass_model = LassoCV()
    lin_model = LinearRegression()
    log_model = LogisticRegression()
    reg_models = [lass_model , lin_model]

    df = get_data()
    df = clean_data(df)
    Classifications = False
    # set X and y
    if not Classifications:
        y = df["fare_amount"]
        X = df.drop(["fare_amount", "fare_amount_catagory"], axis=1)
        # hold out
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

        for reg_model in reg_models :
            tr = Trainer(X_train,y_train, reg_model)
            # build pipeline
            pipeline = tr.set_pipeline()

            # train the pipeline
            tr.run()
            tr.save_model()

            # evaluate the pipeline
            rmse = tr.evaluate(X_val, y_val)
            tr.mlflow_log_metric('rmse',rmse)
            tr.mlflow_log_param('model', reg_model)
            tr.mlflow_log_param("student_name", 'Morad')
        experiment_id = tr.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
    if Classifications :

        y = df["fare_amount_catagory"]
        X = df.drop(["fare_amount", "fare_amount_catagory"], axis=1)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
        tr = Trainer(X_train,y_train, log_model)
        # build pipeline
        pipeline = tr.set_pipeline()

        # train the pipeline
        tr.run()
        tr.save_model()

        # evaluate the pipeline
        accuracy = tr.evaluateClassification(X_val, y_val)
        tr.mlflow_log_metric('accuracy', accuracy)
        tr.mlflow_log_param('model', log_model)
        tr.mlflow_log_param("task", 'Classification')
        tr.mlflow_log_param("student_name", 'Morad')

        experiment_id = tr.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout, Bidirectional
from tensorflow.keras.models import save_model, load_model
from data import Data
import numpy as np
import pickle
import mlflow

class Model_Prediction:

    def __init__(self, crypto_name=""):
        self.data= Data()
        print("crypto name : ", crypto_name)
        self.liste = self.data.getDatasCrypto(crypto_name=crypto_name)

        self.X_train = None
        self.X_test = None
        self.X_dev = None

        self.y_train = None
        self.y_test = None
        self.y_dev = None
        print("##### Creation model OK #####")


    def prepare_data(self):

        # split a univariate sequence into samples
        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence)-1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)
        
 
        # define input sequence
        raw_seq = self.liste
        # choose a number of time steps
        self.n_steps = 10
        # split into samples
        X, y = split_sequence(raw_seq, self.n_steps)

        # Supposons que X et y soient déjà définis à partir de la fonction split_sequence

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        # Afficher les dimensions des ensembles de données
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        #Normaliser les données d'entrée X_train et X_test
        self.X_train = self.scaler_x.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        self.X_test = self.scaler_x.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
        self.X_dev = self.scaler_x.transform(X_dev.reshape(-1, 1)).reshape(X_dev.shape)

        # Normaliser les données de sortie y_train et y_test
        self.y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        self.y_test = self.scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        self.y_dev = self.scaler_y.transform(y_dev.reshape(-1, 1)).reshape(y_dev.shape)

        # Sauvegarder les scalers avec pickle
        with open('scaler_x.pkl', 'wb') as file:
            pickle.dump(self.scaler_x, file)

        with open('scaler_y.pkl', 'wb') as file:
            pickle.dump(self.scaler_y, file)

        with open('X_test.pkl', 'wb') as file:
            pickle.dump(self.X_test, file)

        with open('X_dev.pkl', 'wb') as file:
            pickle.dump(self.X_dev, file)

        with open('y_test.pkl', 'wb') as file:
            pickle.dump(self.y_test, file)

        with open('y_dev.pkl', 'wb') as file:
            pickle.dump(self.y_dev, file)

    
    #entrainement du modèle
    def train_model(self): 
        regressor = Sequential()
        regressor.add(Bidirectional(LSTM(units=150, return_sequences=True, input_shape = (self.X_train.shape[1],1) ) ))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units= 100 , return_sequences=True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units= 80 , return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units= 100))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = self.n_steps, activation='linear'))
        regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
        regressor.fit(self.X_train, self.y_train, epochs=20,batch_size=64 )
        
        self.model = regressor

    # Sauvegarder le model
    def save_model(self):
        print('save_model')
        save_model(self.model, './regressor.h5')

    # charger le model
    def load_model(self):
        print('save_model')
        self.model = load_model('./regressor.h5')

        with open('scaler_x.pkl', 'rb') as file:
            self.scaler_x = pickle.load( file)

        with open('scaler_y.pkl', 'rb') as file:
            self.scaler_y = pickle.load(file)

        with open('X_test.pkl', 'rb') as file:
            self.X_test = pickle.load(file)

        with open('X_dev.pkl', 'rb') as file:
            self.X_dev = pickle.load(file)

        with open('y_test.pkl', 'rb') as file:
            self.y_test = pickle.load(file)

        with open('y_dev.pkl', 'rb') as file:
            self.y_dev = pickle.load(file)




    def evaluate(self):
        evaluation_results = self.model.evaluate(self.X_test, self.y_test)

        print("Loss on test data:", evaluation_results[0])
        print("Accuracy on test data:", evaluation_results[1])

        # Enregistrez les métriques dans MLflow
        with mlflow.start_run():
            mlflow.log_metric("loss", evaluation_results[0])
            mlflow.log_metric("accuracy", evaluation_results[1])

    def save_model(self, model_path):
        # Sauvegardez le modèle dans un chemin spécifié
        self.model.save(model_path)

        # Enregistrez le modèle dans MLflow
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, "model")

    def mlflow_start(self):
        # Commencez l'exécution MLflow
        with mlflow.start_run():
            # Entraînez votre modèle
            self.train_model()

            # Évaluez et enregistrez les résultats
            self.evaluate()

            # Sauvegardez le modèle et enregistrez-le dans MLflow
            model_path = "chemin/vers/votre/repertoire"
            self.save_model(model_path)

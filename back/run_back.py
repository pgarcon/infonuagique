from model import Model_Prediction

model = Model_Prediction(crypto_name='ETH')

#model.prepare_data()
#model.train_model()
#model.save_model()
model.load_model()
model.evaluate()


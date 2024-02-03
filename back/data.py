import pandas as pd

class Data:

    def __init__(self):
        datas = pd.read_csv('datas/datas.csv', delimiter='\t')

        headers = datas.columns.drop('Time')

        datas['Day'] = datas.Time.apply(lambda x: x.split('-')[-1])
        datas['Month'] = datas.Time.apply(lambda x: x.split('-')[1])
        datas['Year'] = datas.Time.apply(lambda x: x.split('-')[0])


        self.liste_crypto_df = {}
        self.liste_title = []
        for crypto in headers:
            df = pd.DataFrame(datas[crypto])

            title = crypto.split(' ')[0]
            self.liste_title.append(title)
            self.liste_crypto_df[title] = df


    ########
    #
    #Retourne la liste des cryptos
    def getDatas(self):
        return self.liste_title
    

    ########
    #
    #Retourne la liste des valeurs pour une crypto donnée
    def getDatasCrypto(self, crypto_name=""):
        print("crypto name to get : ", crypto_name)
        return self.liste_crypto_df[crypto_name].values



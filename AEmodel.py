#ML/DL learning
import pandas as pd 
import numpy as np
import tensorflow as tf

import math



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

# visualisations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')


# misc
import random as rn

# load the dataset
# df = pd.read_csv('../input/creditcard.csv')

# manual parameters
RANDOM_SEED = 42
TRAINING_SAMPLE = 300
VALIDATE_SIZE = 0.2

# setting random seeds for libraries to ensure reproducibility
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)

def flowrate_model(df):
    """Simple flowrate model python function."""
    dnum = 2.0 * math.pi * df[2] * (df[4] - df[5])
    dlnronrw = math.log(df[1] / df[0])
    dden = dlnronrw * (1.0 + (2.0 * df[6] * df[2]) / (dlnronrw * df[0] * df[0] * df[7])
                       + df[2] / df[3])

    return dnum / dden

class EnsembleDenoisingAE():
    def __init__(self):

        self.e1 = 0
        self.e2 = 0.2
        self.e3 = 0.5
        self.e4 = 0.7
        self.e5 = 1

        self.AE1 = None
        self.AE2 = None
        self.AE3 = None
        self.AE4 = None
        self.AE5 = None

        self.Pipe = None
        self.threshold = None

    def train(self,x:pd.DataFrame, e):

        x = x.sample(frac=1).reset_index(drop=True)

        X_train = x.iloc[:TRAINING_SAMPLE]

        X_test = x.iloc[TRAINING_SAMPLE:]

        X_train, X_validate = train_test_split(X_train, 
                                        test_size=VALIDATE_SIZE, 
                                        random_state=RANDOM_SEED)
        

        pipeline = Pipeline([('normalizer', Normalizer()),
                        ('scaler', MinMaxScaler())])
        
        
        pipeline.fit(X_train)

        self.Pipe = pipeline


        X_train_transformed = pipeline.transform(X_train)
        X_validate_transformed = pipeline.transform(X_validate)


        input_dim = X_train_transformed.shape[1]
        BATCH_SIZE = 256
        EPOCHS = 350
        TRAINING_EPOCHS = 40
        TOTAL_BATCH = int(len(X_train_transformed) / BATCH_SIZE)

        model = AutoEncoder(input_dim,e=e)

        optimizer = tf.keras.optimizers.Adam()


        model.compile(optimizer, loss='mse', metrics=["acc"])

        early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.0001,
        patience=500,
        verbose=1, 
        mode='min',
        restore_best_weights=True
        )

        model.fit(
        X_train_transformed, X_train_transformed,
        shuffle=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        validation_data=(X_validate_transformed, X_validate_transformed)
        )

        return model


    def build(self,x:pd.DataFrame):

        x = x.sample(frac=1).reset_index(drop=True)

        X_test = x.iloc[TRAINING_SAMPLE:]

        self.AE1 = self.train(x, self.e1)
        self.AE2 = self.train(x, self.e2)
        self.AE3 = self.train(x, self.e3)
        self.AE4 = self.train(x, self.e4)
        self.AE5 = self.train(x, self.e5)

        print(" All models trained")

        pipeline = Pipeline([('normalizer', Normalizer()),
                        ('scaler', MinMaxScaler())])
        
        pipeline.fit(X_test)


        X_test_transformed = pipeline.transform(X_test)

        reconstruc = self.predict(X_test_transformed)

        mse = np.mean(np.power(X_test_transformed - reconstruc, 2), axis=1)

        print(mse)
        mean = np.mean(mse)
        std = np.std(mse)
        threshold = mean + 2 * std

        self.threshold = threshold

        print("Default threshold fixed")

    def predict(self,x):

        reconstructions1 = self.AE1.predict(x)
        reconstructions2 = self.AE2.predict(x)
        reconstructions3 = self.AE3.predict(x)
        reconstructions4 = self.AE4.predict(x)
        reconstructions5 = self.AE5.predict(x)

        all_reconstructions = [reconstructions1,reconstructions2,reconstructions3,reconstructions4, reconstructions5]

        stacked_r = np.stack(all_reconstructions, axis=0)

        mean_r = np.mean(stacked_r, axis=0)

        print(mean_r)

        return mean_r




class AutoEncoder(tf.keras.Model):
    def __init__(self, n_input, e):
        super(AutoEncoder, self).__init__()

        self.input_dim = n_input 

        self.encoding_1 = tf.keras.layers.Dense(self.input_dim, activation=tf.nn.elu,input_shape=(self.input_dim, ), name = "encoding_1")
        #self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.encoding_2 = tf.keras.layers.Dense(int((self.input_dim/(2**2))*3**3), activation=tf.nn.elu, name = "encoding_2")
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.encoding_3 = tf.keras.layers.Dense(int((self.input_dim/(2**2))*2**3), activation=tf.nn.elu ,name = "encoding_3")
        self.dropout3 = tf.keras.layers.Dropout(0.1)
        self.encoding_4 = tf.keras.layers.Dense(int((self.input_dim/(2**2))*1**3), activation=tf.nn.elu ,name = "encoding_4")
        self.dropout4 = tf.keras.layers.Dropout(0.1)
        self.encoding_final = tf.keras.layers.Dense(2, activation=tf.nn.elu ,name = "encoding_final")
        self.decoding_1 = tf.keras.layers.Dense(int((self.input_dim/(2**2))*1**3), activation=tf.nn.elu ,name = "decoding_1")
        self.dropout5 = tf.keras.layers.Dropout(0.1)
        self.decoding_2 = tf.keras.layers.Dense(int((self.input_dim/(2**2))*2**3), activation=tf.nn.elu ,name = "decoding_2")
        self.dropout6 = tf.keras.layers.Dropout(0.1)
        self.decoding_3 = tf.keras.layers.Dense(int((self.input_dim/(2**2))*3**3), activation=tf.nn.elu ,name = "decoding_3")
        self.dropout7 = tf.keras.layers.Dropout(0.1)
        self.decoding_final = tf.keras.layers.Dense(self.input_dim ,name = "decoding_final")


        self.latent_space = None

        self.noise = e


    # Building the encoder
    def encoder(self,x):
        #x = self.flatten_layer(x)
        layer_1 = self.encoding_1(x)
        #dropout1 = self.dropout1(layer_1)
        layer_2 = self.encoding_2(layer_1)
        dropout2 = self.dropout2(layer_2)
        layer_3 = self.encoding_3(dropout2)
        dropout3 = self.dropout3(layer_3)
        layer_4 = self.encoding_4(dropout3)
        dropout4 = self.dropout4(layer_4)
        encode = self.encoding_final(dropout4)
        return encode
        

    # Building the decoder
    def decoder(self, x):
        layer_1 = self.decoding_1(x)
        dropout1 = self.dropout5(layer_1)
        layer_2 = self.decoding_2(dropout1)
        dropout2 = self.dropout6(layer_2)
        layer_3 = self.decoding_3(dropout2)
        dropout3 = self.dropout7(layer_3)
        
        decode = self.decoding_final(dropout3)
        return decode

        
    def call(self, x):
        encoder_op  = self.encoder(x)
        self.latent_space = encoder_op
        # Reconstructed Images
        y_pred = self.decoder(encoder_op)
        return y_pred
        
def cost(y_true, y_pred):
    # loss = np.mean(np.square(y_true - y_pred))
    loss = tf.keras.losses.MeanSquaredError().call(y_true, y_pred)
    cost = tf.reduce_mean(loss)
    return cost

def grad(self, model, inputs, targets):
    #print('shape of inputs : ',inputs.shape)
    #targets = flatten_layer(targets)
    inputs_noise = inputs + math.sqrt((10^(-0.487/10)))* np.random.normal(0, self.e, size=inputs.shape)
    with tf.GradientTape() as tape:    
        reconstruction = model(inputs_noise)
        loss_value = cost(targets, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables),reconstruction


def AE_model_training(x:pd.DataFrame):

    x = x.sample(frac=1).reset_index(drop=True)

    X_train = x.iloc[:TRAINING_SAMPLE]

    X_test = x.iloc[TRAINING_SAMPLE:]

    X_train, X_validate = train_test_split(X_train, 
                                       test_size=VALIDATE_SIZE, 
                                       random_state=RANDOM_SEED)
    

    pipeline = Pipeline([('normalizer', Normalizer()),
                     ('scaler', MinMaxScaler())])
    
    pipeline.fit(X_train)

    X_train_transformed = pipeline.transform(X_train)
    X_validate_transformed = pipeline.transform(X_validate)


    input_dim = X_train_transformed.shape[1]
    BATCH_SIZE = 256
    EPOCHS = 350
    TRAINING_EPOCHS = 40
    TOTAL_BATCH = int(len(X_train_transformed) / BATCH_SIZE)

    model = AutoEncoder(input_dim, 0)

    optimizer = tf.keras.optimizers.Adam()


    model.compile(optimizer, loss='mse', metrics=["acc"])

    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0.0001,
    patience=500,
    verbose=1, 
    mode='min',
    restore_best_weights=True
    )

    model.fit(
    X_train_transformed, X_train_transformed,
    shuffle=True,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    validation_data=(X_validate_transformed, X_validate_transformed)
    )


    # transform the test set with the pipeline fitted to the training set
    X_test_transformed = pipeline.transform(X_test)

    # pass the transformed test set through the autoencoder to get the reconstructed result
    reconstructions = model.predict(X_test_transformed)

    mse = np.mean(np.power(X_test_transformed - reconstructions, 2), axis=1)

    print(mse)
    mean = np.mean(mse)
    std = np.std(mse)
    threshold = mean + 2 * std


    print("Threshold for anomaly detection", threshold)

    # for epoch in range(TRAINING_EPOCHS):
    #     for i in range(TOTAL_BATCH):
    #         x_inp = X_train_transformed[i : i + BATCH_SIZE]
    #         loss_value, grads, reconstruction = grad(model, x_inp, x_inp)
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     # Display logs per epoch step
    #     if epoch % 1 == 0:
    #         print("Epoch:", '%04d' % (epoch+1),
    #             "cost=", "{:.9f}".format(loss_value))

    print("Optimization Finished!")

    return model, threshold, pipeline, mean, std

def calculate_sensitivity(X_test_transformed, reconstructions, threshold_conf=2):
    """
    Calculate the sensitivity of each parameter for the AutoEncoder.

    Parameters:
    - X_test_transformed: ndarray, the normalized and scaled test data.
    - reconstructions: ndarray, the reconstructed data from the AutoEncoder.
    - threshold_conf: float, confidence multiplier for threshold calculation (default: 2).

    Returns:
    - sensitivities: ndarray, sensitivity values for each parameter.
    - alarms: ndarray, binary alarm indicator for each parameter (1: alarm, 0: no alarm).
    """

    # Step 1: Calculate Residuals
    residuals = np.abs(X_test_transformed - reconstructions)  # Element-wise residuals
    max_residuals = np.max(residuals, axis=0)  # Max residual for each parameter

    # Step 2: Define Threshold for Anomalies
    mse = np.mean(np.power(X_test_transformed - reconstructions, 2), axis=1)
    mean = np.mean(mse)
    std = np.std(mse)
    threshold = mean + threshold_conf * std

    # Step 3: Calculate Critical Functions (Flattened KDE)
    sensitivities = []
    alarms = []

    for param_idx in range(X_test_transformed.shape[1]):
        # Extract residuals for the current parameter
        param_residuals = residuals[:, param_idx]
        print(param_residuals)
        # KDE for the residuals
        kde = gaussian_kde(param_residuals, bw_method='scott')
        grid = np.linspace(param_residuals.min() * 0.8, param_residuals.max() * 1.2, 100)
        pdf = kde(grid)

        # Define the critical function (flattening tails)
        cdf = np.cumsum(pdf) / np.sum(pdf)
        critical_value_idx = np.argmax(cdf >= 0.95)  # Critical value at 95% confidence
        critical_point = grid[critical_value_idx]

        # Flattened critical function
        y_lower = 0.8 * grid.min()
        y_upper = 1.2 * grid.max()
        critical_func = interp1d(grid, grid <= critical_point, bounds_error=False, fill_value=(critical_point, critical_point))

        # Sensitivity Calculation
        max_g = critical_func(max_residuals[param_idx])
        sensitivity = max_residuals[param_idx] / max_g - 1

        sensitivities.append(sensitivity)
        alarms.append(int(sensitivity > 0))  # Raise alarm if sensitivity > 0

    return np.array(sensitivities), np.array(alarms)

if __name__ == '__main__':

    df = pd.read_csv('train_corr4.txt',delimiter='\t')

    # print(df.columns.tolist())
    # columns_to_drop = ['date ', 'chu ', 'chl ', 'cr  ','nLoop ','end ']
    # df.drop(columns=columns_to_drop, inplace=True)
    # df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)   
    # print(df)

    train = df.iloc[:,1:]


    model, threshold, pipeline, mean, std = AE_model_training(train)


    unlabelled_test = pd.read_csv('train_corr4(copie 1).txt',delimiter='\t')

    #print(unlabelled_test)

    x_unlabelled_test = unlabelled_test.iloc[:, 0]

    y_unlabelled_test = unlabelled_test.iloc[:, 1:]



    encode_re = model.encoder(y_unlabelled_test)

    PCA_transformer1 = PCA(n_components=0.95)
    print(train)

    pipe1 = Pipeline([('scaler', MinMaxScaler()), ('pca',PCA_transformer1)])
    Xt = pipe1.fit_transform(train)
    print(Xt)

    PCA_transformer = PCA(n_components=0.95)
    print(train)

    pipe = Pipeline([('scaler', MinMaxScaler()), ('pca',PCA_transformer)])
    Xt = pipe.fit_transform(encode_re)
    print(Xt)
    # value_counts = unlabelled_test['Altered'].value_counts()
    # print(value_counts)
    # plot = plt.scatter(Xt[:,0], Xt[:,1], c=x_unlabelled_test)
    # plt.legend(handles=plot.legend_elements()[0], labels=["Normal","Altered"])
    # plt.show()

    print("Variance expliquée : ", PCA_transformer.explained_variance_ratio_)

    

    




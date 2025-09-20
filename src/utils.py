import zipfile
import os
import shutil
import logging
import sys
import pickle
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def unzip_file(source: str,
    dest: str):
    """
    Unzip a file
    """
    try:
        for filename in os.listdir(source):
            if filename.endswith('.zip'):
                zip_path = os.path.join(source, filename)
                extract_file(source=zip_path, dest=dest)
                logging.info(f"File unzip completed for: {filename}")
                    
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(f"Error unzipping file: {custom_error}")
        raise
    

def clean_reshape_files_to_npy(source: str, dest: str):
    """
    We do data wrangling here by:
       1- Drop rows that contain at least one NaN in the columns
       2- Keep only full month users, filter by ACCOUNT_IDENTIFIER
       3- Reshape the data, rows are time intervals and columns are users
       4- Save the reshaped data as numpy array (npy)
    """
    try:
        for filename in os.listdir(source):
            if filename.endswith('.csv'):
                file_path = os.path.join(source, filename)
                df = pd.read_csv(file_path)

                # remove rows with missing values
                time_intervally_cols = df.iloc[:, 7:55]
                df = df[~time_intervally_cols.isna().any(axis=1)]

                # Add DAY variable for latter chronologial sorting
                df['INTERVAL_READING_DATE'] = pd.to_datetime(df['INTERVAL_READING_DATE'])
                df['DAY'] = df['INTERVAL_READING_DATE'].dt.day

                # keep only full months users
                user_date_counts = (
                    df.groupby('ACCOUNT_IDENTIFIER')['INTERVAL_READING_DATE']
                    .count()
                    .reset_index(name='DATE_COUNT')
                )
                full_month_users = (
                    user_date_counts
                    .query('DATE_COUNT == 31')
                    ['ACCOUNT_IDENTIFIER']
                    .tolist()

                )
                df = (
                    df[df.ACCOUNT_IDENTIFIER.isin(full_month_users)]
                )
                # Keep only full time users
                reshaped_user_data = []
                for user in full_month_users:
                    user_df = (
                        df.query('ACCOUNT_IDENTIFIER == @user')
                        .sort_values('DAY')
                    )
                    reshaped_user_data.append(
                        user_df[list_time_interval_columns()].values.reshape(-1, 1)
                    )
                reshaped_user_data = np.hstack(reshaped_user_data)

                save_npy(
                    filename=filename[:-4],
                    dest=dest,
                    arr=reshaped_user_data
                    )
                logging.info(f"Data wrangling completed for: {filename}")
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(f"Error data wrangling is: {custom_error}")
        raise

def RBD_method(data: np.ndarray, col: int, tol: float = 1e-5) -> np.ndarray:     

    try:   

        data = data.astype(float)
        # Praparation of the algorithm
        (nr, nc) = data.shape
        bases = np.zeros((nr, col), dtype=float)
        TransMat = np.zeros((col, nc), dtype=float)
        AtAAtXi = np.zeros((nc, col), dtype=float)
        xiFlag = np.zeros(col, dtype=int)
        xiFlag[0] = np.random.randint(nc)
        xiFlag[0] = 1
        i = 0
        CurErr = tol + 1

        # Preparation for efficient error evaluation
        ftf = np.sum(data.T * data.T, axis=1)

        # The RBD greedy algorithm
        while (i < col) and (CurErr > tol):
            biCand = data[:, xiFlag[i]]
            # Inside: Gram-Schmidt orthonormalization of the current candidate with
            # all previsouly chosen basis vectors
            for j in np.arange(i):
                biCand = biCand - np.dot(biCand.T, bases[:, j]) * bases[:, j]
            normi = np.linalg.norm(biCand)

            if normi < 1e-7:
                print("Reduced system getting singular - to stop with", i - 1, "basis functions")
                bases = bases[:, :i - 1]
                TransMat = TransMat[:i - 1, :]
                break
            else:
                bases[:, i] = (biCand / normi).flatten()

            TransMat[i, :] = np.dot(bases[:, i].T, data)

            # Inside: With one more basis added, we need to update what allows for
            # the efficient error evaluation.
            AtAAtXi[:, i] = np.dot(data.T, bases[:, i])
            # Inside: Efficiently go through all the columns to identify where the
            # error would be the largest if we were to use the current space for compression.

            TMM = TransMat[:i+1, :]
            te1 = np.sum(AtAAtXi[:, :i+1] * TMM.T, axis=1)
            te2 = np.sum(TMM.T * TMM.T, axis=1)
            errord = ftf - 2 * te1 + te2
            CurErr = np.max(errord)
            TempPos = np.argmax(errord)

            # Mark this location for the next round
            if (i < col - 1):
                xiFlag[i + 1] = TempPos
            # If the largest error is small enough, we announce and stop.
            if CurErr <= tol:
                print(CurErr)
                print("Reduced system getting singular - to stop with", i - 1, "basis functions")
                bases = bases[:, :i + 1]
                TransMat = TransMat[:i + 1, :]
            else:
                i += 1

        return bases[:, :col]
    
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(f"Error in RBD: {custom_error}")
        raise


def PCA_method(data: np.ndarray, dim_reduction_size: int) -> np.ndarray:
    try:
        data_scaled = StandardScaler().fit_transform(data)
        pca = PCA()
        coeff = pca.fit(data_scaled).components_.T
        score = pca.transform(data_scaled)
        # data_pca_full = pca.fit_transform(data_scaled)
        # data_pca_reduced = data_pca_full[:, :dim_reduction_size]
        data_pca_reduced = data_scaled @ coeff[:, :dim_reduction_size]
        return data_pca_reduced, pca
    
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(f"Error in PCA: {custom_error}")
        raise


def customize_time_interval(time_interval: int, data: np.ndarray) -> np.ndarray:
    try:
        step = int(2*time_interval)
        row_size, col_size = int(data.shape[0] // step), int(data.shape[1])
        data_intervals=np.zeros((row_size, col_size))

        for i in range(step):
            data_intervals += data[i::step, :]
        return data_intervals
        
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(f"Error in customize_time_interval: {custom_error}")
        raise



def make_data_r_time_intervals_dependent(data: np.ndarray, r: int) -> np.ndarray:
    """
    For the output, the first r columns are the r previous time intervals
    and the last column is the current time interval which is used as dependent
    variable (y) later. We convert a matrix of shape (m, n) to a matrix of shape (m-r, r*n+1) 
    """

    try:
        data_with_previous_r_intervals = []
        for i in range(data.shape[1]):
            col_with_r_intervals = []
            for j in range(data.shape[0]-r):
                col_with_r_intervals.append(data[j:j+r, i].reshape(1, -1))
            col_with_r_intervals = np.vstack(col_with_r_intervals)
            data_with_previous_r_intervals.append(col_with_r_intervals)
        return np.hstack(data_with_previous_r_intervals)

    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(f"Error in make_data_r_time_intervals_dependent: {custom_error}")
        raise



def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            # train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(f"Error in evaluate_models: {custom_error}")



def extract_file(source: str, dest: str):
    try:
        with zipfile.ZipFile(source, 'r') as zip_ref:
            os.makedirs(dest, exist_ok=True)
            zip_ref.extractall(dest)
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(custom_error)
        raise


def save_npy(filename: str, dest: str, arr):
    try:
        file_path = os.path.join(dest, filename)
        os.makedirs(dest, exist_ok=True)
        np.save(file_path, arr)
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(custom_error)
        raise


def load_npy(filename: str, source: str):
    try:
        file_path = os.path.join(source, filename)
        return np.load(file_path)
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(custom_error)
        raise


def save_object(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(custom_error)
        raise
    

def list_time_interval_columns() -> list[str]:
    time_interval_columns = [
    f"INTERVAL_HR{(30 * i) // 60:02d}{(30 * i) % 60:02d}_ENERGY_QTY"
    for i in range(1, 49)
    ]
    return time_interval_columns


def estimate_mean_absolute_percentage(y_true, y_pred):
    try:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        e_map = np.mean(np.abs((y_true - y_pred) / y_true))
        return e_map
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(custom_error)
        raise


def estimate_coefficient_of_error(y_true, y_pred):
    try:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        e_cv = np.std(y_true - y_pred) / np.mean(y_true)
        return e_cv
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(custom_error)
        raise

def relative_error(y_true, y_pred):
    try:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        relative_error = (y_true - y_pred) / y_true
        return relative_error
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(custom_error)
        raise
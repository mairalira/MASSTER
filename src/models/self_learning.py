import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import time
from sklearn.ensemble import RandomForestRegressor

def get_datasets(i):
    df_train_labeled = pd.read_csv(r'C:\Users\danin\Documents\lu\mestrado\artigo_ricardo\Active-and-Self-Learning-for-Multi-target-Regression\src\processed\atp7d\train_'+str(i+1))
    df_train_unlabeled = pd.read_csv(r'C:\Users\danin\Documents\lu\mestrado\artigo_ricardo\Active-and-Self-Learning-for-Multi-target-Regression\src\processed\atp7d\pool_'+str(i+1))
    df_test = pd.read_csv(r'C:\Users\danin\Documents\lu\mestrado\artigo_ricardo\Active-and-Self-Learning-for-Multi-target-Regression\src\processed\atp7d\test_'+str(i+1))

    X_test_labeled=df_test.drop(columns = ['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000'])
    y_test_labeled = df_test[['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000']]

    X_unlabeled = df_train_unlabeled.drop(columns = ['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000'])
    X_train_labeled = df_train_labeled.drop(columns = ['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000'])
    y_train_labeled = df_train_labeled[['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000']]
    Y_train_missing = df_train_unlabeled[['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000']]
    Y_train_missing.loc[:, :] = np.nan  # Substitui todos os valores do DataFrame por np.nan
    
    X_train_labeled = X_train_labeled.to_numpy()
    y_train_labeled = y_train_labeled.to_numpy()
    X_unlabeled = X_unlabeled.to_numpy()
    X_test_labeled = X_test_labeled.to_numpy()
    y_test_labeled = y_test_labeled.to_numpy()

        
    
    return X_train_labeled,y_train_labeled,X_unlabeled,X_test_labeled,y_test_labeled,Y_train_missing


# Função de acurácia personalizada
def custom_accuracy(y_true, y_pred, threshold=0.1):
    """Custom accuracy: percentage of predictions within a certain threshold."""
    return np.mean(np.abs(y_true - y_pred) <= threshold)

# Carregar os dados


number_of_pools = 10
for i in range (number_of_pools):
    print(f"Treinando modelo no pool {i}...")  
    X_train_not_missing, Y_train_not_missing, X_unlabeled, X_test_labeled, y_test_labeled, Y_train_missing = get_datasets(i)

# Avaliação do desempenho original com o modelo treinado com os dados completos (X_train_not_missing e Y_train_not_missing)
    print("Desempenho com Dados Originais (sem aprendizado ativo):")
    start_time = time.time()  # Medir o tempo de execução
    regressor = RandomForestRegressor()

    regressor.fit(X_train_not_missing, Y_train_not_missing)
    Y_pred_original = regressor.predict(X_test_labeled)
    for j in range(y_test_labeled.shape[1]):
        r2 = r2_score(y_test_labeled[:, j], Y_pred_original[:, j])
        mae = mean_absolute_error(y_test_labeled[:, j], Y_pred_original[:, j])
        ca = custom_accuracy(y_test_labeled[:, j], Y_pred_original[:, j])
        print(f"Target {j+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")
    print(f"Tempo de execução (com dados originais): {time.time() - start_time:.2f} segundos\n")


    print("Desempenho Semissupervisionado")
    # Aprendizado Ativo
    max_iter = 10
    for j in range(max_iter):  
        print(f"Treinando modelo na epoch {j}...")  
        # modelo de regressão treinado com dados rotulados 
        predictions_unlabeled = regressor.predict(X_unlabeled)
        error_threshold = np.percentile(np.abs(predictions_unlabeled - Y_train_missing), 90)
        confident_idx = np.where(np.abs(predictions_unlabeled - Y_train_missing) < error_threshold)[0]
        
        # Adicionar os dados de alta confiança ao conjunto de treinamento
        X_train_not_missing = np.concatenate([X_train_not_missing, X_unlabeled[confident_idx]], axis=0)
        Y_train_not_missing = np.concatenate([Y_train_not_missing, predictions_unlabeled[confident_idx]], axis=0)
        
        # Atualizar o conjunto de dados não rotulado
        X_unlabeled = np.delete(X_unlabeled, confident_idx, axis=0)
        Y_train_missing = np.delete(Y_train_missing, confident_idx, axis=0)
        regressor.fit(X_train_not_missing, Y_train_not_missing)
    # Avaliação final
    regressor.fit(X_train_not_missing, Y_train_not_missing)
    Y_pred_semi_weighted = regressor.predict(X_test_labeled)
    for j in range(y_test_labeled.shape[1]):
        r2 = r2_score(y_test_labeled[:, j], Y_pred_semi_weighted[:, j])
        mae = mean_absolute_error(y_test_labeled[:, j], Y_pred_semi_weighted[:, j])
        ca = custom_accuracy(y_test_labeled[:, j], Y_pred_semi_weighted[:, j])
        print(f"Target {i+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")

    # Tempo de execução final
    end_time = time.time()
    print(f"Tempo de execução total: {end_time - start_time:.2f} segundos")

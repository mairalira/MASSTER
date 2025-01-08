from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def get_datasets(i):
    df_train_labeled = pd.read_csv(r'C:\Users\danin\Documents\lu\mestrado\artigo_ricardo\Active-and-Self-Learning-for-Multi-target-Regression\src\processed\atp7d\train_'+str(i))
    df_train_unlabeled = pd.read_csv(r'C:\Users\danin\Documents\lu\mestrado\artigo_ricardo\Active-and-Self-Learning-for-Multi-target-Regression\src\processed\atp7d\pool_'+str(i))
    df_test = pd.read_csv(r'C:\Users\danin\Documents\lu\mestrado\artigo_ricardo\Active-and-Self-Learning-for-Multi-target-Regression\src\processed\atp7d\test_'+str(i))

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

# Função de precisão personalizada
def custom_accuracy(y_true, y_pred, threshold=0.1):
    """Custom accuracy: percentage of predictions within a certain threshold."""
    return np.mean(np.abs(y_true - y_pred) <= threshold)


number_of_pools = 10
for i in range (number_of_pools):
    print(f"Treinando modelo no pool {i}...")
    X_train_not_missing, Y_train_not_missing, X_unlabeled, X_test_labeled, y_test_labeled, Y_train_missing = get_datasets(i+1)

    print("Desempenho Original:")
    start_time = time.time()

        # Inicializar um modelo de regressão utilizando os dados rotulados completos
    model_original = MultiOutputRegressor(RandomForestRegressor(random_state=42))

    model_original.fit(X_train_not_missing, Y_train_not_missing)

        # Fazer previsões nos dados de teste
    Y_pred_original = model_original.predict(X_test_labeled)
    execution_time = time.time() - start_time
    for j in range(y_test_labeled.shape[1]):
        r2 = r2_score(y_test_labeled[:, j], Y_pred_original[:, j])
        mae = mean_absolute_error(y_test_labeled[:, j], Y_pred_original[:, j])
        ca = custom_accuracy(y_test_labeled[:, j], Y_pred_original[:, j])
        print(f"Target {i+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")
    print(f"Tempo de execução (com dados originais): {time.time() - start_time:.2f} segundos\n")



    print("Desempenho Semissupervisionado")
    
    # Modelos para regressão multi-target
    print("Inicializando modelos de regressão...")

    model_view1 = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model_view2 = MultiOutputRegressor(RandomForestRegressor(random_state=42))
     # Dividir as features em duas visões
    print("Dividindo as features em duas visões...")
        # Dividindo as features corretamente
    X_train_labeled_v1 = X_train_not_missing[:, :int(X_train_not_missing.shape[1]/2)]  # Primeira metade das características
    X_train_labeled_v2 = X_train_not_missing[:, int(X_train_not_missing.shape[1]/2):]  # Segunda metade das características
        # Agora, aplique a máscara para filtrar X_train_not_missing e X_view2
    X_train_unlabeled_v1 = X_unlabeled[:, :int(X_train_not_missing.shape[1]/2)]
    X_train_unlabeled_v2 = X_unlabeled[:, int(X_train_not_missing.shape[1]/2):]

    X_test_labeled_v1 = X_test_labeled[:, :int(X_test_labeled.shape[1] / 2)]  # Primeira metade das características
    X_test_labeled_v2 = X_test_labeled[:, int(X_test_labeled.shape[1] / 2):]  # Segunda metade das características
        
    y_labeled = Y_train_not_missing
    max_iter = 10
    for j in range (max_iter):
        print(f"Treinando modelo na epoch {j}...")  

        # Treinamento com co-training
        start_time = time.time()
        print("Iniciando treinamento com co-training...")
        
        
        print("Treinando modelo 1 com visão 1...")
        model_view1.fit(X_train_labeled_v1, y_labeled)
        print("Treinando modelo 2 com visão 2...")
        model_view2.fit(X_train_labeled_v2, y_labeled)

        # Fazer previsões para os dados não rotulados
        print("Fazendo previsões para dados não rotulados...")

        preds1 = model_view1.predict(X_train_unlabeled_v1) if len(X_train_unlabeled_v1) > 0 else np.array([])
        preds2 = model_view2.predict(X_train_unlabeled_v2) if len(X_train_unlabeled_v2) > 0 else np.array([])

        if len(preds1) == 0 or len(preds2) == 0:
            print("Sem mais exemplos não rotulados.")
            break

        threshold=0.1

        # Determinar confiança das previsões com base na variação
        print("Calculando a confiança das previsões...")
        confident_mask1 = np.std(preds1, axis=1) <= threshold
        confident_mask2 = np.std(preds2, axis=1) <= threshold
        # Adicionar exemplos confiáveis ao conjunto rotulado
        if confident_mask1.any() and confident_mask2.any():
            print(f"Adicionando {confident_mask1.sum()} exemplos confiáveis da visão 1.")
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_train_unlabeled_v1[confident_mask1]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_train_unlabeled_v2[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds1[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds2[confident_mask2]])

        if not confident_mask1.any() and not confident_mask2.any():
            print("Nenhuma previsão confiável encontrada.")
            break

        # Adicionar exemplos confiáveis ao conjunto rotulado
        if confident_mask1.any():
            print(f"Adicionando {confident_mask1.sum()} exemplos confiáveis da visão 1.")
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_train_unlabeled_v1[confident_mask1]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_train_unlabeled_v2[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds1[confident_mask1]])

        if confident_mask2.any():
            print(f"Adicionando {confident_mask2.sum()} exemplos confiáveis da visão 2.")
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_train_unlabeled_v1[confident_mask2]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_train_unlabeled_v2[confident_mask2]])
            y_labeled = np.vstack([y_labeled, preds2[confident_mask2]])

        # Remover exemplos confiáveis do conjunto não rotulado
        print("Removendo exemplos confiáveis do conjunto não rotulado...")
        #checar se tá sendo atualizado 
        X_train_unlabeled_v1 = X_train_unlabeled_v1[~confident_mask1]
        X_train_unlabeled_v2 = X_train_unlabeled_v2[~confident_mask2]

        print(f"{confident_mask1.sum() + confident_mask2.sum()} exemplos adicionados nesta iteração.")    
        
        # Avaliação nos dados de teste
    print("Fazendo previsões nos dados de teste...")
    y_pred_v1 = model_view1.predict(X_test_labeled_v1)
    y_pred_v2 = model_view2.predict(X_test_labeled_v2)
    execution_time = time.time() - start_time
        # Combinar previsões por média
    y_pred_combined = (y_pred_v1 + y_pred_v2) / 2

    # Avaliação do desempenho semissupervisionado
    for j in range(y_test_labeled.shape[1]):  # Iterando por cada target
            r2 = r2_score(y_test_labeled[:, j], y_pred_combined[:, j])
            mae = mean_absolute_error(y_test_labeled[:, j], y_pred_combined[:, j])
            ca = custom_accuracy(y_test_labeled[:, j], y_pred_combined[:, j])
            print(f"Target {i+1}: R²={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")

    print(f"Tempo de execução: {execution_time:.2f} segundos\n")

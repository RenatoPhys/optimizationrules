# CURVA ROC - ANÁLISE DE DESEMPENHO
print("\n" + "="*60)
print("CURVA ROC - RECEIVER OPERATING CHARACTERISTIC")
print("="*60)

from sklearn.metrics import roc_curve, auc

# Calculando as probabilidades para treino e teste
y_proba_train = modelo.predict_proba(X_train)[:, 1]
y_proba_test_roc = modelo.predict_proba(X_test)[:, 1]

# Convertendo labels para binário (1 para Default, 0 para Não Default)
y_train_binary = (y_train == 'Sim').astype(int)
y_test_binary = (y_test == 'Sim').astype(int)

# Calculando a curva ROC para treino e teste
fpr_train, tpr_train, thresholds_train = roc_curve(y_train_binary, y_proba_train)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test_binary, y_proba_test_roc)

# Calculando AUC (Area Under Curve)
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

print(f"\nAUC (Area Under Curve):")
print(f"  - Treino: {auc_train:.4f}")
print(f"  - Teste: {auc_test:.4f}")
print(f"  - Diferença: {abs(auc_train - auc_test):.4f}")

# Plotando as curvas ROC
plt.figure(figsize=(10, 8))

# Curva ROC do treino
plt.plot(fpr_train, tpr_train, 'b-', linewidth=2, 
         label=f'ROC Treino (AUC = {auc_train:.3f})')

# Curva ROC do teste
plt.plot(fpr_test, tpr_test, 'r-', linewidth=2, 
         label=f'ROC Teste (AUC = {auc_test:.3f})')

# Linha diagonal (classificador aleatório)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Classificador Aleatório')

# Destacando alguns pontos importantes
# Encontrando o ponto ótimo (mais próximo do canto superior esquerdo)
optimal_idx = np.argmax(tpr_test - fpr_test)
optimal_threshold = thresholds_test[optimal_idx]
plt.scatter(fpr_test[optimal_idx], tpr_test[optimal_idx], 
           color='red', s=100, zorder=5)
plt.annotate(f'Ponto Ótimo\n(threshold={optimal_threshold:.2f})', 
            xy=(fpr_test[optimal_idx], tpr_test[optimal_idx]),
            xytext=(fpr_test[optimal_idx] + 0.1, tpr_test[optimal_idx] - 0.1),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC - Comparação Treino vs Teste')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Análise de interpretação
print("\n" + "-"*60)
print("INTERPRETAÇÃO DA CURVA ROC:")
print("-"*60)

if auc_test >= 0.9:
    classificacao = "EXCELENTE"
    descricao = "O modelo tem poder discriminatório excepcional"
elif auc_test >= 0.8:
    classificacao = "MUITO BOM"
    descricao = "O modelo tem ótima capacidade de distinguir defaults"
elif auc_test >= 0.7:
    classificacao = "BOM"
    descricao = "O modelo tem boa performance, adequado para produção"
elif auc_test >= 0.6:
    classificacao = "REGULAR"
    descricao = "O modelo tem performance limitada, considere melhorias"
else:
    classificacao = "FRACO"
    descricao = "O modelo precisa ser reformulado"

print(f"\nClassificação do modelo: {classificacao}")
print(f"Descrição: {descricao}")

if abs(auc_train - auc_test) > 0.05:
    print("\n⚠️ ATENÇÃO: Diferença significativa entre treino e teste!")
    print("   Possível overfitting. Considere:")
    print("   - Reduzir a profundidade da árvore")
    print("   - Aumentar min_samples_split e min_samples_leaf")
    print("   - Usar validação cruzada para ajustar hiperparâmetros")
else:
    print("\n✓ Boa generalização: performance similar em treino e teste")
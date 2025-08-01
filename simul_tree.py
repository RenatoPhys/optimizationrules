# Simulação de Árvore de Decisão para Classificação
# Exemplo: Previsão de Default (Inadimplência) em Empréstimos

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração para melhor visualização
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# 1. CRIANDO O DATASET
print("="*60)
print("1. CRIANDO DATASET DE CLIENTES COM EMPRÉSTIMO")
print("="*60)

# Criando dados sintéticos mais realistas
np.random.seed(42)  # Para reprodutibilidade

# Gerando 300 exemplos de clientes que já têm empréstimo aprovado
n_samples = 300

# Características do cliente
renda = np.random.choice(['Baixa', 'Média', 'Alta'], n_samples, p=[0.35, 0.45, 0.20])
historico_credito = np.random.choice(['Ruim', 'Regular', 'Bom'], n_samples, p=[0.20, 0.40, 0.40])
tempo_emprego = np.random.choice(['< 1 ano', '1-5 anos', '> 5 anos'], n_samples, p=[0.25, 0.45, 0.30])
idade = np.random.choice(['18-25', '26-40', '41-60', '> 60'], n_samples, p=[0.20, 0.40, 0.30, 0.10])

# Características do empréstimo
valor_emprestimo = np.random.choice(['Baixo', 'Médio', 'Alto'], n_samples, p=[0.30, 0.50, 0.20])
prazo = np.random.choice(['6 meses', '12 meses', '24 meses', '36 meses'], n_samples, p=[0.15, 0.35, 0.35, 0.15])
taxa_juros = np.random.choice(['Baixa', 'Normal', 'Alta'], n_samples, p=[0.25, 0.50, 0.25])

# Criando a lógica de default (com algum ruído para tornar mais realista)
default = []
for i in range(n_samples):
    risco = 0
    
    # Fatores que aumentam o risco de default
    if renda[i] == 'Baixa':
        risco += 3
    elif renda[i] == 'Média':
        risco += 1
    
    if historico_credito[i] == 'Ruim':
        risco += 4
    elif historico_credito[i] == 'Regular':
        risco += 2
    
    if tempo_emprego[i] == '< 1 ano':
        risco += 3
    elif tempo_emprego[i] == '1-5 anos':
        risco += 1
    
    if idade[i] == '18-25':
        risco += 2
    elif idade[i] == '> 60':
        risco += 1
    
    if valor_emprestimo[i] == 'Alto':
        risco += 3
    elif valor_emprestimo[i] == 'Médio':
        risco += 1
    
    if prazo[i] == '36 meses':
        risco += 2
    elif prazo[i] == '24 meses':
        risco += 1
    
    if taxa_juros[i] == 'Alta':
        risco += 2
    elif taxa_juros[i] == 'Normal':
        risco += 1
    
    # Decisão com algum ruído
    if risco >= 10:
        default.append('Sim' if np.random.random() > 0.15 else 'Não')
    elif risco >= 6:
        default.append('Sim' if np.random.random() > 0.5 else 'Não')
    elif risco >= 3:
        default.append('Sim' if np.random.random() > 0.8 else 'Não')
    else:
        default.append('Não' if np.random.random() > 0.1 else 'Sim')

# Criando o DataFrame
df = pd.DataFrame({
    'Renda': renda,
    'Histórico_Crédito': historico_credito,
    'Tempo_Emprego': tempo_emprego,
    'Idade': idade,
    'Valor_Empréstimo': valor_emprestimo,
    'Prazo': prazo,
    'Taxa_Juros': taxa_juros,
    'Default': default
})

print("\nPrimeiras 10 linhas do dataset:")
print(df.head(10))

print(f"\nTotal de clientes com empréstimo: {len(df)}")
print(f"Clientes que deram default: {(df['Default'] == 'Sim').sum()} ({(df['Default'] == 'Sim').sum()/len(df)*100:.1f}%)")
print(f"Clientes adimplentes: {(df['Default'] == 'Não').sum()} ({(df['Default'] == 'Não').sum()/len(df)*100:.1f}%)")

# Análise exploratória rápida
print("\n" + "="*60)
print("ANÁLISE EXPLORATÓRIA - Taxa de Default por Categoria")
print("="*60)

# Taxa de default por renda
print("\nTaxa de default por faixa de renda:")
for renda_cat in ['Baixa', 'Média', 'Alta']:
    taxa = (df[df['Renda'] == renda_cat]['Default'] == 'Sim').mean() * 100
    print(f"  {renda_cat}: {taxa:.1f}%")

# Taxa de default por histórico
print("\nTaxa de default por histórico de crédito:")
for hist in ['Ruim', 'Regular', 'Bom']:
    taxa = (df[df['Histórico_Crédito'] == hist]['Default'] == 'Sim').mean() * 100
    print(f"  {hist}: {taxa:.1f}%")

# 2. PREPARANDO OS DADOS
print("\n" + "="*60)
print("2. PREPARANDO OS DADOS PARA O MODELO")
print("="*60)

# Convertendo variáveis categóricas em numéricas
df_encoded = pd.get_dummies(df, columns=['Renda', 'Histórico_Crédito', 'Tempo_Emprego', 
                                         'Idade', 'Valor_Empréstimo', 'Prazo', 'Taxa_Juros'])
print("\nTotal de variáveis após codificação:", len(df_encoded.columns) - 1)

# Separando features (X) e target (y)
X = df_encoded.drop('Default', axis=1)
y = df['Default']

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nDados de treino: {len(X_train)} exemplos")
print(f"Dados de teste: {len(X_test)} exemplos")

# 3. TREINANDO A ÁRVORE DE DECISÃO
print("\n" + "="*60)
print("3. TREINANDO A ÁRVORE DE DECISÃO")
print("="*60)

# Criando e treinando o modelo
modelo = DecisionTreeClassifier(
    max_depth=5,  # Limitando a profundidade para evitar overfitting
    min_samples_split=15,  # Mínimo de amostras para dividir um nó
    min_samples_leaf=8,  # Mínimo de amostras em uma folha
    class_weight='balanced',  # Importante quando temos desbalanceamento de classes
    random_state=42
)

modelo.fit(X_train, y_train)
print("✓ Modelo treinado com sucesso!")

# 4. VISUALIZANDO A ÁRVORE
print("\n" + "="*60)
print("4. VISUALIZANDO A ÁRVORE DE DECISÃO")
print("="*60)

# Plotando a árvore
plt.figure(figsize=(20, 12))
plot_tree(modelo, 
          feature_names=X.columns,
          class_names=['Não Default', 'Default'],
          filled=True,
          rounded=True,
          fontsize=9,
          proportion=True)  # Mostra proporções em vez de contagens
plt.title("Árvore de Decisão - Previsão de Default em Empréstimos", fontsize=16)
plt.tight_layout()
plt.show()

# Versão simplificada da árvore (primeiros níveis)
print("\nPrimeiras regras da árvore (profundidade 3):")
modelo_simples = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo_simples.fit(X_train, y_train)
print(export_text(modelo_simples, feature_names=list(X.columns), max_depth=3))

# 5. AVALIANDO O MODELO
print("\n" + "="*60)
print("5. AVALIANDO O DESEMPENHO DO MODELO")
print("="*60)

# Fazendo previsões
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)
y_proba_test = modelo.predict_proba(X_test)

# Acurácia
print(f"\nAcurácia no treino: {accuracy_score(y_train, y_pred_train)*100:.2f}%")
print(f"Acurácia no teste: {accuracy_score(y_test, y_pred_test)*100:.2f}%")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_test, labels=['Não', 'Sim'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Default', 'Default'], 
            yticklabels=['Não Default', 'Default'])
plt.title('Matriz de Confusão - Previsão de Default')
plt.ylabel('Valor Real')
plt.xlabel('Previsão')

# Adicionando métricas na matriz
total = np.sum(cm)
for i in range(2):
    for j in range(2):
        plt.text(j + 0.5, i + 0.7, f'{cm[i,j]/total*100:.1f}%',
                ha='center', va='center', fontsize=10, color='red')
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_test, 
                          target_names=['Não Default', 'Default']))

# Análise de custo (muito importante para default)
print("\n" + "="*60)
print("ANÁLISE DE CUSTO DO MODELO")
print("="*60)

# Definindo custos hipotéticos
custo_falso_negativo = 10000  # Não prever um default que acontece
custo_falso_positivo = 500    # Prever default que não acontece

tn, fp, fn, tp = cm.ravel()
custo_total = (fn * custo_falso_negativo) + (fp * custo_falso_positivo)

print(f"\nCustos assumidos:")
print(f"  - Falso Negativo (não detectar default): R$ {custo_falso_negativo:,}")
print(f"  - Falso Positivo (alarme falso): R$ {custo_falso_positivo:,}")
print(f"\nResultados:")
print(f"  - Falsos Negativos: {fn} (custo: R$ {fn * custo_falso_negativo:,})")
print(f"  - Falsos Positivos: {fp} (custo: R$ {fp * custo_falso_positivo:,})")
print(f"  - Custo total estimado: R$ {custo_total:,}")

# 6. IMPORTÂNCIA DAS FEATURES
print("\n" + "="*60)
print("6. IMPORTÂNCIA DAS VARIÁVEIS")
print("="*60)

# Calculando importância
importancia = pd.DataFrame({
    'Feature': X.columns,
    'Importância': modelo.feature_importances_
}).sort_values('Importância', ascending=False)

print("\nTop 10 variáveis mais importantes:")
print(importancia.head(10).to_string(index=False))

# Visualizando importância
plt.figure(figsize=(10, 8))
top_features = importancia.head(15)
plt.barh(range(len(top_features)), top_features['Importância'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importância')
plt.title('Top 15 Variáveis Mais Importantes para Prever Default')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 7. SIMULANDO ANÁLISE DE RISCO
print("\n" + "="*60)
print("7. SIMULANDO ANÁLISE DE RISCO PARA NOVOS CLIENTES")
print("="*60)

# Criando novos exemplos para teste
novos_clientes = pd.DataFrame({
    'Renda': ['Alta', 'Baixa', 'Média', 'Baixa'],
    'Histórico_Crédito': ['Bom', 'Ruim', 'Regular', 'Regular'],
    'Tempo_Emprego': ['> 5 anos', '< 1 ano', '1-5 anos', '> 5 anos'],
    'Idade': ['41-60', '18-25', '26-40', '26-40'],
    'Valor_Empréstimo': ['Baixo', 'Alto', 'Médio', 'Baixo'],
    'Prazo': ['12 meses', '36 meses', '24 meses', '12 meses'],
    'Taxa_Juros': ['Baixa', 'Alta', 'Normal', 'Normal']
})

print("\nPerfil dos novos clientes:")
for i, (idx, cliente) in enumerate(novos_clientes.iterrows()):
    print(f"\nCliente {i+1}:")
    for col, val in cliente.items():
        print(f"  {col}: {val}")

# Preparando os dados
novos_encoded = pd.get_dummies(novos_clientes)
# Alinhando com as colunas do treino
novos_encoded = novos_encoded.reindex(columns=X.columns, fill_value=0)

# Fazendo previsões
previsoes = modelo.predict(novos_encoded)
probabilidades = modelo.predict_proba(novos_encoded)

print("\n" + "-"*60)
print("RESULTADO DA ANÁLISE DE RISCO:")
print("-"*60)

for i in range(len(novos_clientes)):
    risco_default = probabilidades[i][1] * 100
    classificacao = "ALTO RISCO" if risco_default > 50 else "MÉDIO RISCO" if risco_default > 25 else "BAIXO RISCO"
    
    print(f"\nCliente {i+1}:")
    print(f"  → Probabilidade de default: {risco_default:.1f}%")
    print(f"  → Classificação: {classificacao}")
    print(f"  → Recomendação: {'Monitorar de perto' if previsoes[i] == 'Sim' else 'Cliente confiável'}")

# 8. ANÁLISE DE DIFERENTES THRESHOLDS
print("\n" + "="*60)
print("8. ANÁLISE DE DIFERENTES PONTOS DE CORTE")
print("="*60)

# Calculando métricas para diferentes thresholds
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
resultados = []

for thresh in thresholds:
    y_pred_thresh = (y_proba_test[:, 1] >= thresh).astype(int)
    y_pred_thresh = ['Sim' if x == 1 else 'Não' for x in y_pred_thresh]
    
    precisao = precision_score(y_test, y_pred_thresh, pos_label='Sim')
    recall = recall_score(y_test, y_pred_thresh, pos_label='Sim')
    f1 = f1_score(y_test, y_pred_thresh, pos_label='Sim')
    
    resultados.append({
        'Threshold': thresh,
        'Precisão': precisao,
        'Recall': recall,
        'F1-Score': f1
    })

df_thresh = pd.DataFrame(resultados)
print("\nMétricas por threshold:")
print(df_thresh.to_string(index=False))

# Visualizando
plt.figure(figsize=(10, 6))
plt.plot(df_thresh['Threshold'], df_thresh['Precisão'], 'o-', label='Precisão', linewidth=2)
plt.plot(df_thresh['Threshold'], df_thresh['Recall'], 'o-', label='Recall', linewidth=2)
plt.plot(df_thresh['Threshold'], df_thresh['F1-Score'], 'o-', label='F1-Score', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Trade-off entre Precisão e Recall para Diferentes Thresholds')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9. CONCLUSÕES E RECOMENDAÇÕES
print("\n" + "="*60)
print("9. CONCLUSÕES E RECOMENDAÇÕES PARA O NEGÓCIO")
print("="*60)

print("""
INSIGHTS DO MODELO:

1. PRINCIPAIS INDICADORES DE RISCO:
   - Histórico de crédito ruim é o maior preditor de default
   - Clientes com menos de 1 ano de emprego têm maior risco
   - Empréstimos de valor alto com prazo longo são mais arriscados

2. PERFIL DE CLIENTE DE ALTO RISCO:
   - Renda baixa + Histórico ruim + Emprego recente
   - Jovens (18-25) com empréstimos altos
   - Taxa de juros alta (pode indicar risco já precificado)

3. RECOMENDAÇÕES OPERACIONAIS:
   - Implementar monitoramento mensal para clientes de alto risco
   - Oferecer renegociação preventiva para reduzir defaults
   - Ajustar políticas de crédito baseadas nos insights

4. OTIMIZAÇÃO DO MODELO:
   - Considerar usar Random Forest para melhor performance
   - Coletar mais variáveis (ex: score de crédito, região)
   - Ajustar threshold baseado no apetite de risco do negócio

5. CONSIDERAÇÕES ÉTICAS:
   - Evitar discriminação por idade ou região
   - Transparência nas decisões de crédito
   - Dar oportunidades de melhoria do perfil de risco
""")

# EXERCÍCIOS PROPOSTOS
print("\n" + "="*60)
print("EXERCÍCIOS PARA OS ALUNOS")
print("="*60)

print("""
1. ANÁLISE CRÍTICA:
   - O modelo está sendo justo com todos os grupos?
   - Que outras variáveis poderiam melhorar a previsão?
   - Como você explicaria uma negação de crédito para um cliente?

2. EXPERIMENTOS:
   - Mude o threshold e veja o impacto no lucro/prejuízo
   - Remova a variável mais importante e avalie o impacto
   - Crie um modelo apenas com 3 variáveis

3. APLICAÇÃO PRÁTICA:
   - Calcule o ROI de implementar este modelo
   - Proponha uma estratégia de cobrança baseada no risco
   - Desenvolva um score de 0-100 baseado na probabilidade

4. DESAFIO AVANÇADO:
   - Implemente validação cruzada temporal
   - Compare com regressão logística
   - Crie um ensemble de modelos
""")
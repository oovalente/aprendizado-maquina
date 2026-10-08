"""Gera e executa o notebook da prática 6."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# Prática 6 - Regressão logística e SHAP
**Valentim Soares de Andrade Neto | 2019027210**

Objetivo: classificar tumores do Breast Cancer Wisconsin (Diagnostic), avaliar o modelo em teste e explicar suas previsões. Rótulos originais: **0 = maligno, 1 = benigno**. Não há ajuste de hiperparâmetros usando o teste.
''')
code('''from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn, shap
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from IPython.display import display
OUT = Path('output'); FIG = OUT / 'figuras'; FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')
def salvar(nome):
    plt.savefig(FIG / (nome + '.png'), dpi=170, bbox_inches='tight')
    plt.show(); plt.close()
dados = load_breast_cancer()
X = pd.DataFrame(dados.data, columns=dados.feature_names)
y = pd.Series(dados.target, index=X.index, name='diagnosis')
display(X.head())
print(X.shape, 'Ausentes:', X.isna().sum().sum(), 'Duplicados:', X.duplicated().sum())
display(y.value_counts().rename(index={0: 'Maligno', 1: 'Benigno'}))
display(X.describe().T)
X.describe().T.to_csv(OUT / 'resumo_estatistico.csv')
''')
md('''## 1. Exploração
São 569 amostras, 30 atributos numéricos e duas classes. Os atributos são derivados de imagens de punção aspirativa por agulha fina: raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal, com média, erro padrão e “worst” (média dos três maiores valores). Não há necessidade de imputação. Os gráficos exploratórios usam o conjunto completo apenas para descrição; o pré-processamento e o modelo usam somente o treino.
''')
code('''fig, axes = plt.subplots(2, 3, figsize=(12, 7))
features = ['mean radius', 'mean texture', 'mean area', 'mean concavity', 'mean concave points', 'worst radius']
for ax, feature in zip(axes.flat, features):
    sns.boxplot(x=y.map({0:'Maligno', 1:'Benigno'}), y=X[feature], ax=ax)
    ax.set(xlabel='', ylabel=feature)
fig.suptitle('Distribuições por diagnóstico'); plt.tight_layout(); salvar('distribuicoes')
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlação entre os 30 atributos'); plt.tight_layout(); salvar('correlacoes')
''')
md('''Raio, perímetro e área representam aspectos relacionados ao tamanho e são fortemente correlacionados. As classes mostram diferenças nas distribuições, com sobreposição. Essa redundância deve ser considerada ao interpretar a importância individual dos atributos.

## 2. Divisão, padronização e treinamento
Reproduzimos a divisão do exemplo: 80%/20%, `random_state=42`, sem estratificação. O StandardScaler aprende média e desvio padrão somente nos 455 registros de treino; o teste contém 114 registros. A regressão logística usa regularização L2, C=1, solver lbfgs e máximo de 2.000 iterações. P(maligno) corresponde à coluna da classe 0.
''')
code('''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
scaler = StandardScaler()
Z_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
Z_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
modelo = LogisticRegression(C=1, solver='lbfgs', max_iter=2000, random_state=42)
modelo.fit(Z_train, y_train)
pred = modelo.predict(Z_test)
prob_maligno = modelo.predict_proba(Z_test)[:, list(modelo.classes_).index(0)]
acc = accuracy_score(y_test, pred)
auc = roc_auc_score((y_test == 0).astype(int), prob_maligno)
cm = confusion_matrix(y_test, pred, labels=[0, 1])
print('Acurácia:', acc, 'AUC (maligno positivo):', auc)
print(classification_report(y_test, pred, target_names=['Maligno', 'Benigno']))
display(pd.crosstab(y, pd.Series(np.where(X.index.isin(X_train.index), 'Treino', 'Teste'), index=X.index)))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Maligno','Benigno'], yticklabels=['Maligno','Benigno'], ax=axes[0])
axes[0].set(xlabel='Predição', ylabel='Diagnóstico real', title='Matriz de confusão')
RocCurveDisplay.from_predictions((y_test == 0).astype(int), prob_maligno, ax=axes[1], name='Maligno')
axes[1].plot([0,1],[0,1], '--', color='gray'); plt.tight_layout(); salvar('avaliacao')
''')
md('''## 3. Explicações SHAP globais
O LinearExplainer com masker Independent usa todos os registros padronizados de treino como referência. Explica f(x) = log(P(benigno)/P(maligno)), e não a probabilidade diretamente. SHAP positivo favorece benigno; negativo favorece maligno. A probabilidade benigna é sigmoid(f(x)); a maligna é 1 - sigmoid(f(x)). O valor base é a média dos **log-odds** no treino, diferente do logit da probabilidade média.

Sob essa referência independente, φ_j = coef_j × (z_j - média_treino_j). Verificamos a soma aditiva para todas as amostras de teste e a equivalência com as probabilidades. As correlações não são repartidas entre os atributos: o ranking descreve este modelo e esta referência, sem demonstrar causalidade.
''')
code('''masker = shap.maskers.Independent(Z_train, max_samples=len(Z_train))
explainer = shap.LinearExplainer(modelo, masker)
sv = explainer(Z_test)
scores = modelo.decision_function(Z_test)
erro = float(np.max(np.abs(sv.base_values + sv.values.sum(axis=1) - scores)))
assert np.allclose(sv.base_values + sv.values.sum(axis=1), scores)
assert np.allclose(1-expit(scores), prob_maligno)
assert np.allclose(sv.values, (Z_test.to_numpy()-Z_train.mean().to_numpy())*modelo.coef_[0])
ranking = pd.DataFrame({'variavel': X.columns, 'media_abs_shap': np.abs(sv.values).mean(axis=0), 'coeficiente': modelo.coef_[0]}).sort_values('media_abs_shap', ascending=False)
display(ranking.head(10)); ranking.to_csv(OUT / 'importancia_shap.csv', index=False)
shap.summary_plot(sv.values, Z_test, plot_type='bar', max_display=15, show=False)
plt.title('Importância global: média de |SHAP| em log-odds'); salvar('shap_bar')
shap.summary_plot(sv.values, Z_test, max_display=15, show=False)
plt.title('Direção das contribuições para benigno'); salvar('shap_beeswarm')
''')
md('''## 4. Três pacientes do teste
São usados os três primeiros registros na ordem devolvida pela divisão, sem seleção baseada no resultado. As barras vermelhas aumentam os log-odds de benignidade; as azuis os reduzem. Os valores apresentados junto aos atributos estão padronizados. A tabela exportada conserva os valores originais.
''')
code('''pacientes = []
for i in range(3):
    idx = int(X_test.index[i]); order = np.argsort(-np.abs(sv.values[i]))[:4]
    registro = {'indice': idx, 'real': 'Maligno' if y_test.iloc[i]==0 else 'Benigno', 'predito': 'Maligno' if pred[i]==0 else 'Benigno', 'p_maligno': float(prob_maligno[i]), 'base': float(sv.base_values[i]), 'score': float(scores[i]), 'principais': [{'variavel': str(X.columns[j]), 'valor_original':float(X_test.iloc[i,j]), 'shap':float(sv.values[i,j])} for j in order]}
    pacientes.append(registro)
    print(f"Paciente {i+1}, índice {idx}: real={registro['real']}, predito={registro['predito']}, P(maligno)={prob_maligno[i]:.4%}, f(x)={scores[i]:.4f}")
    display(pd.DataFrame(registro['principais']))
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    plt.title(f'Paciente {i+1} (índice {idx}) - log-odds de benigno'); salvar(f'paciente_{i+1}')
    print('Principais contribuições:', '; '.join(f"{r['variavel']}: {r['shap']:+.3f}, favorece " + ('benigno' if r['shap']>0 else 'maligno') for r in registro['principais']))
pd.DataFrame(sv.values, index=X_test.index, columns=X.columns).to_csv(OUT / 'shap_teste.csv', index_label='indice')
X_test.to_csv(OUT / 'atributos_teste.csv', index_label='indice')
pd.DataFrame({'real':y_test, 'predito':pred, 'p_maligno':prob_maligno}).to_csv(OUT / 'predicoes.csv', index_label='indice')
resultados = {'acuracia':float(acc), 'auc_maligno':float(auc), 'matriz_confusao':cm.tolist(), 'sensibilidade_maligno':float(cm[0,0]/cm[0].sum()), 'especificidade':float(cm[1,1]/cm[1].sum()), 'treino':len(X_train), 'teste':len(X_test), 'classes':{str(k):int(v) for k,v in y.value_counts().items()}, 'base':float(sv.base_values[0]), 'erro_aditividade':erro, 'ranking':ranking.head(10).to_dict('records'), 'pacientes':pacientes, 'versoes':{'python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__, 'sklearn':sklearn.__version__, 'shap':shap.__version__}}
(OUT / 'resultados.json').write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding='utf-8')
''')
md('''## 5. Discussão e referências
O ranking calculado acima permite identificar os atributos de maior impacto médio no teste. As explicações locais mostram por que a mesma variável pode contribuir de modos diferentes conforme o valor observado. Características de tamanho e irregularidade têm plausibilidade na discriminação de tumores; o sinal deve ser lido junto ao coeficiente e ao valor padronizado, considerando a redundância entre medições. SHAP não estabelece efeito causal nem garante generalização fora desta amostra.

A acurácia e AUC avaliam aspectos distintos: decisões no limiar 0,5 e ordenação dos riscos em vários limiares. Os erros da matriz de confusão permanecem relevantes mesmo com AUC elevada. O estudo utiliza uma única divisão, não avalia calibração nem validação externa e tem finalidade didática.

- [Dataset e atributos - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [LinearExplainer - SHAP](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html)
- [Waterfall - SHAP](https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html)
''')

if __name__ == '__main__':
    nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb, ROOT / 'shap_regressao_logistica.ipynb')
    print('Notebook executado e resultados exportados.')

"""Gera o notebook executado, os graficos e o relatorio da PT-03."""

from pathlib import Path
import json
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
DATA = ROOT / "tips.csv"
NOTEBOOK = ROOT / "regressao_linear_manual.ipynb"
REPORT = ROOT / "2019027210_Valentim_S_A_Neto_Pratica3.pdf"


def ajustar_regressao_linear(x, y):
    """Calcula beta0 e beta1 por minimos quadrados, sem estimador pronto."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("x e y devem ter o mesmo tamanho e ao menos 2 observacoes")
    n = x.size
    denominador = n * np.sum(x ** 2) - np.sum(x) ** 2
    if np.isclose(denominador, 0):
        raise ValueError("nao e possivel ajustar: x nao possui variacao")
    beta1 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / denominador
    beta0 = (np.sum(y) - beta1 * np.sum(x)) / n
    return float(beta0), float(beta1)


def prever(x, beta0, beta1):
    return beta0 + beta1 * np.asarray(x, dtype=float)


def calcular_metricas(y_real, y_pred):
    """Retorna RSE (com n-2 graus de liberdade) e R2."""
    y_real = np.asarray(y_real, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_real.size != y_pred.size or y_real.size <= 2:
        raise ValueError("as metricas exigem vetores iguais com mais de 2 valores")
    sse = np.sum((y_real - y_pred) ** 2)
    rse = np.sqrt(sse / (y_real.size - 2))
    sst = np.sum((y_real - np.mean(y_real)) ** 2)
    r2 = 1 - sse / sst
    return float(rse), float(r2)


def carregar_tips():
    if not DATA.exists():
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
        urllib.request.urlretrieve(url, DATA)
    return pd.read_csv(DATA)


def grafico_parte1(x_train, y_train, x_test, y_test, b0, b1):
    grade = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_train, y_train, label="Treino", color="#2563eb", s=48)
    ax.scatter(x_test, y_test, label="Teste", color="#f59e0b", marker="D", s=55)
    ax.plot(grade, prever(grade, b0, b1), label="Reta ajustada no treino", color="#dc2626", lw=2)
    ax.set(title="Regressao linear manual: altura e peso", xlabel="Altura (m)", ylabel="Peso (kg)")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    caminho = ASSETS / "parte1_altura_peso.png"
    fig.savefig(caminho, dpi=180); plt.close(fig)
    return caminho


def grafico_parte2(x_train, y_train, x_test, y_test, b0, b1):
    ordem = np.argsort(x_test)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_train, y_train, label="Treino", color="#2563eb", alpha=.68, s=28)
    ax.scatter(x_test, y_test, label="Teste (real)", color="#f59e0b", marker="D", s=42)
    ax.plot(x_test[ordem], prever(x_test[ordem], b0, b1), label="Previsao no teste", color="#dc2626", lw=2.2)
    ax.set(title="Regressao linear manual: total da conta e gorjeta", xlabel="Total da conta (US$)", ylabel="Gorjeta (US$)")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    caminho = ASSETS / "parte2_tips.png"
    fig.savefig(caminho, dpi=180); plt.close(fig)
    return caminho


X1 = np.array([1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70,
               1.73, 1.75, 1.78, 1.80, 1.83])
y1 = np.array([52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29,
               63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46])
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=.2, random_state=42)
b01, b11 = ajustar_regressao_linear(X1_train, y1_train)
pred1 = prever(X1_test, b01, b11)
rse1, r21 = calcular_metricas(y1_test, pred1)
plot1 = grafico_parte1(X1_train, y1_train, X1_test, y1_test, b01, b11)

tips = carregar_tips()
X2 = tips["total_bill"].to_numpy()
y2 = tips["tip"].to_numpy()
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=.2, random_state=42)
b02, b12 = ajustar_regressao_linear(X2_train, y2_train)
pred2 = prever(X2_test, b02, b12)
rse2, r22 = calcular_metricas(y2_test, pred2)
gorjeta80 = float(prever(80, b02, b12))
plot2 = grafico_parte2(X2_train, y2_train, X2_test, y2_test, b02, b12)


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source, output=None, count=None):
    outputs = [] if output is None else [{"name": "stdout", "output_type": "stream", "text": [output]}]
    return {"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outputs,
            "source": source.splitlines(True)}


common_code = '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def ajustar_regressao_linear(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = x.size
    denominador = n * np.sum(x**2) - np.sum(x)**2
    if np.isclose(denominador, 0):
        raise ValueError("x precisa possuir variacao")
    beta1 = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / denominador
    beta0 = (np.sum(y) - beta1*np.sum(x)) / n
    return beta0, beta1

def prever(x, beta0, beta1):
    return beta0 + beta1*np.asarray(x, dtype=float)

def calcular_metricas(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    sse = np.sum((y_real-y_pred)**2)
    rse = np.sqrt(sse/(len(y_real)-2))
    r2 = 1 - sse/np.sum((y_real-y_real.mean())**2)
    return rse, r2
'''

part1_code = '''X = np.array([1.47,1.50,1.52,1.55,1.57,1.60,1.63,1.65,1.68,1.70,1.73,1.75,1.78,1.80,1.83])
y = np.array([52.21,53.12,54.48,55.84,57.20,58.57,59.93,61.29,63.11,64.47,66.28,68.10,69.92,72.19,74.46])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
beta0,beta1=ajustar_regressao_linear(X_train,y_train)
y_pred=prever(X_test,beta0,beta1)
rse,r2=calcular_metricas(y_test,y_pred)
print(f"beta0={beta0:.6f}; beta1={beta1:.6f}; RSE={rse:.6f}; R2={r2:.6f}")
grade=np.linspace(X.min(),X.max(),200)
plt.scatter(X_train,y_train,label="Treino"); plt.scatter(X_test,y_test,label="Teste")
plt.plot(grade,prever(grade,beta0,beta1),color="red",label="Reta")
plt.xlabel("Altura (m)"); plt.ylabel("Peso (kg)"); plt.legend(); plt.grid(alpha=.25); plt.show()
'''

part2_code = '''tips=pd.read_csv("tips.csv")
X=tips["total_bill"].to_numpy(); y=tips["tip"].to_numpy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
beta0,beta1=ajustar_regressao_linear(X_train,y_train)
y_pred=prever(X_test,beta0,beta1)
rse,r2=calcular_metricas(y_test,y_pred)
gorjeta80=prever(80,beta0,beta1)
print(f"beta0={beta0:.6f}; beta1={beta1:.6f}; RSE={rse:.6f}; R2={r2:.6f}")
print(f"Previsao para conta de US$ 80: US$ {gorjeta80:.2f}")
ordem=np.argsort(X_test)
plt.scatter(X_train,y_train,label="Treino",alpha=.65); plt.scatter(X_test,y_test,label="Teste")
plt.plot(X_test[ordem],y_pred[ordem],color="red",label="Previsao no teste")
plt.xlabel("Total da conta (US$)"); plt.ylabel("Gorjeta (US$)"); plt.legend(); plt.grid(alpha=.25); plt.show()
'''

cells = [
    md("# PT-03 - Regressao Linear Manual\n\n**Aluno:** Valentim S. A. Neto  \n**Matricula:** 2019027210  \n**Disciplina:** Aprendizado de Maquina\n\nOs coeficientes e as metricas sao calculados manualmente; `train_test_split` e usado somente para separar os dados."),
    md("## Funcoes\n\nA inclinacao e o intercepto seguem as formulas de minimos quadrados do enunciado. O RSE usa `n-2` graus de liberdade e o R2 usa a soma total de quadrados calculada no conjunto de teste."),
    code(common_code, count=1),
    md("## Parte 1 - Altura e peso\n\nComo o enunciado pede avaliacao no teste mas nao especifica a divisao nesta parte, adotamos 80%/20% e `random_state=42`, igual a parte 2."),
    code(part1_code, f"beta0={b01:.6f}; beta1={b11:.6f}; RSE={rse1:.6f}; R2={r21:.6f}\n", 2),
    md(f"**Resultado.** O RSE foi **{rse1:.3f} kg** e o R2 foi **{r21:.4f}**. Neste pequeno conjunto, o modelo explica quase toda a variacao dos pesos de teste."),
    md("## Parte 2 - Dataset tips"),
    code(part2_code, f"beta0={b02:.6f}; beta1={b12:.6f}; RSE={rse2:.6f}; R2={r22:.6f}\nPrevisao para conta de US$ 80: US$ {gorjeta80:.2f}\n", 3),
    md(f"**Resultado.** No teste, RSE = **US$ {rse2:.3f}** e R2 = **{r22:.4f}**. Para uma conta de **US$ 80**, a extrapolacao linear estima uma gorjeta de **US$ {gorjeta80:.2f}**. Como US$ 80 esta acima do maior total observado, essa previsao deve ser interpretada com cautela."),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
executado = nbformat.read(NOTEBOOK, as_version=4)
NotebookClient(
    executado,
    timeout=300,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
).execute()
nbformat.write(executado, NOTEBOOK)


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#17365D")))
styles.add(ParagraphStyle(name="BodyJust", parent=styles["BodyText"], alignment=TA_JUSTIFY, leading=15, spaceAfter=8))
styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8.5, textColor=colors.HexColor("#555555"), leading=11))

def cabecalho_rodape(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2*cm, A4[1]-1.25*cm, "Aprendizado de Maquina - PT-03")
    canvas.drawRightString(A4[0]-2*cm, 1.15*cm, f"Pagina {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(str(REPORT), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2.4*cm, bottomMargin=1.7*cm)
story = [
    Spacer(1, .35*cm), Paragraph("Regressao Linear Manual", styles["TitleCenter"]),
    Paragraph("PT-03 - Relatorio de resultados", styles["Heading2"]),
    Paragraph("Valentim S. A. Neto - Matricula 2019027210", styles["BodyText"]), Spacer(1, .25*cm),
    Paragraph("Objetivo", styles["Heading2"]),
    Paragraph("Implementar uma regressao linear simples sem utilizar estimadores prontos, calcular os coeficientes por minimos quadrados e avaliar as previsoes com o erro padrao residual (RSE) e o coeficiente de determinacao (R2).", styles["BodyJust"]),
    Paragraph("Metodologia", styles["Heading2"]),
    Paragraph("Os dados foram separados em 80% para treino e 20% para teste com random_state=42. A mesma regra foi aplicada a parte 1, pois o enunciado pede metricas no teste sem indicar outra divisao. Os coeficientes beta0 e beta1 foram calculados apenas sobre o treino. O RSE foi calculado como raiz de SSE/(n-2), exatamente como no enunciado; o R2 compara SSE com a variacao dos valores reais do teste.", styles["BodyJust"]),
    Paragraph("Parte 1 - Previsao de peso pela altura", styles["Heading2"]),
    Image(str(plot1), width=15.5*cm, height=9.7*cm),
    Table([["Coeficiente/Metrica", "Valor"], ["beta0", f"{b01:.6f}"], ["beta1", f"{b11:.6f} kg/m"], ["RSE (teste)", f"{rse1:.6f} kg"], ["R2 (teste)", f"{r21:.6f}"]], colWidths=[7*cm, 6*cm], style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17365D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.5,colors.grey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ALIGN",(1,1),(1,-1),"RIGHT"),("PADDING",(0,0),(-1,-1),6)])),
    Spacer(1,.25*cm), Paragraph(f"O baixo RSE e o R2 de {r21:.4f} indicam excelente ajuste aos tres exemplos reservados para teste. Como o conjunto e muito pequeno, o resultado e sensivel a divisao escolhida.", styles["BodyJust"]),
    PageBreak(), Paragraph("Parte 2 - Dataset tips", styles["Heading2"]),
    Paragraph(f"O dataset possui {len(tips)} atendimentos. O modelo usa o total da conta como entrada e o valor da gorjeta como saida.", styles["BodyJust"]),
    Image(str(plot2), width=15.5*cm, height=9.7*cm),
    Table([["Coeficiente/Metrica", "Valor"], ["beta0", f"{b02:.6f}"], ["beta1", f"{b12:.6f}"], ["RSE (teste)", f"US$ {rse2:.6f}"], ["R2 (teste)", f"{r22:.6f}"], ["Previsao para US$ 80", f"US$ {gorjeta80:.2f}"]], colWidths=[7*cm, 6*cm], style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17365D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.5,colors.grey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ALIGN",(1,1),(1,-1),"RIGHT"),("PADDING",(0,0),(-1,-1),6)])),
    Spacer(1,.3*cm), Paragraph(f"O modelo explica {r22*100:.1f}% da variacao das gorjetas no conjunto de teste. A previsao para uma conta de US$ 80 e US$ {gorjeta80:.2f}.", styles["BodyJust"]),
    Paragraph("Cuidado com a extrapolacao", styles["Heading3"]),
    Paragraph(f"O maior total da conta observado no dataset e US$ {tips.total_bill.max():.2f}. Assim, US$ 80 fica fora da faixa dos dados usados no ajuste; o valor solicitado e uma extrapolacao e possui incerteza maior.", styles["BodyJust"]),
    Paragraph("Conclusao", styles["Heading2"]),
    Paragraph("A implementacao manual reproduz todas as etapas essenciais da regressao linear simples: ajuste dos coeficientes no treino, previsao e avaliacao no teste. A relacao altura-peso do exemplo e quase perfeitamente linear. No dataset tips, o total da conta explica parte relevante, mas nao toda, da variabilidade das gorjetas.", styles["BodyJust"]),
    Paragraph("Fonte dos dados: seaborn-data/tips.csv, acessado durante a geracao e preservado localmente no projeto.", styles["SmallNote"]),
]
doc.build(story, onFirstPage=cabecalho_rodape, onLaterPages=cabecalho_rodape)

print(f"Parte 1: beta0={b01:.6f}, beta1={b11:.6f}, RSE={rse1:.6f}, R2={r21:.6f}")
print(f"Parte 2: beta0={b02:.6f}, beta1={b12:.6f}, RSE={rse2:.6f}, R2={r22:.6f}")
print(f"Gorjeta para US$ 80: US$ {gorjeta80:.2f}")
print(f"Notebook: {NOTEBOOK}\nRelatorio: {REPORT}")

"""Gera e executa o notebook da PT-02 e produz o relatorio final em PDF."""

from pathlib import Path
import io
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from nbclient import NotebookClient
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph,
    PageBreak, Spacer, Table, TableStyle,
)
from sklearn.datasets import load_wine


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
BOXPLOTS = ASSETS / "boxplots"
NOTEBOOK = ROOT / "analise_exploratoria_wine.ipynb"
REPORT = ROOT / "2019027210_Valentim_S_A_Neto_Pratica2.pdf"
ASSETS.mkdir(exist_ok=True)
BOXPLOTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
features = list(wine.feature_names)
cor = df[features].corr()

pairs = []
for i, left in enumerate(features):
    for right in features[i + 1:]:
        value = cor.loc[left, right]
        if abs(value) >= 0.8:
            pairs.append((left, right, value))

def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

outliers = {feature: iqr_outliers(df[feature]) for feature in features}
outlier_text = ", ".join(f"{k} ({v})" for k, v in outliers.items() if v)


def markdown(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    markdown("""
    # PT-02 - Análise Exploratória do dataset Wine

    **Aluno:** Valentim S. A. Neto  
    **Matrícula:** 2019027210  
    **Disciplina:** Aprendizado de Máquina

    ## Objetivo

    Investigar as 13 variáveis de entrada do dataset Wine, procurando características
    redundantes ou pouco informativas por meio de estatística descritiva, correlações,
    scatterplots e boxplots. O notebook funciona como uma análise reproduzível; as
    interpretações consolidadas também aparecem no relatório em PDF.
    """),
    markdown("""
    ## 1. Carregamento e identificação das variáveis

    A documentação do `load_wine` informa 178 amostras, 13 entradas numéricas positivas
    e três classes. A saída `target` codifica as classes 0, 1 e 2 (`class_0`, `class_1`
    e `class_2`). Abaixo carregamos os dados diretamente do scikit-learn.

    Fonte: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
    """),
    code("""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_wine

    sns.set_theme(style="whitegrid", context="notebook")
    wine_data = load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df["target"] = wine_data.target

    print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print("Entradas:", ", ".join(wine_data.feature_names))
    print("Saída: target ->", dict(enumerate(wine_data.target_names)))
    df.head()
    """),
    markdown("""
    **Interpretação.** Cada linha representa uma amostra de vinho. As entradas são:
    `alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`,
    `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`,
    `color_intensity`, `hue`, `od280/od315_of_diluted_wines` e `proline`.
    A saída é `target`, uma classe categórica com três valores.
    """),
    markdown("## 2. Descrição estatística\n\n`describe()` resume contagem, média, desvio-padrão, quartis, mínimo e máximo."),
    code("""
    resumo = df.drop(columns="target").describe().T
    resumo.round(2)
    """),
    markdown("""
    **Interpretação.** Todas as entradas têm 178 observações, portanto não há valores
    ausentes. As escalas são muito diferentes: `proline` chega a 1.680, enquanto
    outras variáveis ficam próximas de zero. Essa diferença não invalida a AED, mas
    exigiria padronização antes de muitos algoritmos baseados em distância.
    """),
    code("""
    assert df.shape == (178, 14)
    assert df.isna().sum().sum() == 0
    assert set(df["target"].unique()) == {0, 1, 2}
    assert (df.drop(columns="target") > 0).all().all()
    print("Validações concluídas: dimensões, ausência de nulos, classes e positividade.")
    """),
    markdown("## 3. Matriz de correlações\n\nA correlação de Pearson mede relações lineares. Consideraremos forte uma relação com $|r| \\geq 0{,}80$."),
    code("""
    correlacoes = df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlacoes, annot=True, cmap="coolwarm", center=0,
                vmin=-1, vmax=1, fmt=".2f", annot_kws={"size": 8})
    plt.title("Matriz de correlação - Wine")
    plt.tight_layout()
    plt.show()
    """),
    markdown(f"""
    **Interpretação.** Entre as entradas, apenas `total_phenols` e `flavanoids`
    ultrapassam o limiar do enunciado, com **r = {cor.loc['total_phenols','flavanoids']:.3f}**.
    Elas carregam informação linear bastante semelhante e formam o principal caso de
    redundância. A correlação com `target` ajuda a interpretar associação, mas o alvo é
    categórico e seus códigos são ordinais apenas por convenção; portanto, ela não deve
    ser tratada isoladamente como medida definitiva de relevância.
    """),
    markdown("## 4. Matriz de scatterplots\n\nA cor identifica as classes. A diagonal mostra a distribuição de cada entrada."),
    code("""
    pairplot = sns.pairplot(
        df, hue="target", diag_kind="hist", corner=True,
        plot_kws={"alpha": 0.70, "s": 20, "edgecolor": "none"},
        diag_kws={"alpha": 0.65},
    )
    pairplot.fig.suptitle("Matriz de scatterplots por classe", y=1.01)
    plt.show()
    """),
    markdown("""
    **Interpretação.** `flavanoids`, `od280/od315_of_diluted_wines`, `proline`,
    `color_intensity` e `alcohol` mostram separações visuais úteis em diferentes
    classes. `total_phenols` acompanha fortemente `flavanoids`, reforçando a
    redundância. `ash`, isoladamente, apresenta ampla sobreposição entre as três
    classes; `magnesium` também tem sobreposição considerável. Nenhuma variável,
    sozinha, separa perfeitamente todas as classes.
    """),
    markdown("""
    ## 5. Boxplots individuais e outliers

    Cada variável possui escala própria; por isso, cada boxplot é exibido em um gráfico
    separado. Isso evita que `proline` comprima visualmente as demais distribuições.
    """),
    code("""
    def contar_outliers_iqr(serie):
        q1, q3 = serie.quantile([0.25, 0.75])
        iqr = q3 - q1
        return int(((serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)).sum())

    contagem_outliers = {}
    for feature in wine_data.feature_names:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="target", y=feature, hue="target",
                    palette="Set2", legend=False, ax=ax)
        ax.set_title(f"{feature} por classe")
        ax.set_xlabel("Classe (target)")
        ax.set_ylabel(feature)
        plt.tight_layout()
        plt.show()
        contagem_outliers[feature] = contar_outliers_iqr(df[feature])

    pd.Series(contagem_outliers, name="outliers_IQR").sort_values(ascending=False).to_frame()
    """),
    markdown(f"""
    **Interpretação.** Pelo critério global de Tukey (1,5 vezes o intervalo
    interquartil), há observações atípicas em: {outlier_text}. Um ponto fora dos
    limites do boxplot não é automaticamente erro e não foi removido. As distribuições
    de `ash` são muito sobrepostas; `magnesium` também discrimina pouco isoladamente.
    Já `flavanoids`, `od280/od315_of_diluted_wines`, `proline`, `color_intensity` e
    `alcohol` apresentam deslocamentos claros entre pelo menos algumas classes.
    """),
    markdown(f"""
    ## 6. Conclusão

    1. **Variáveis altamente correlacionadas:** `total_phenols` e `flavanoids`
       (r = {cor.loc['total_phenols','flavanoids']:.3f}). A semelhança linear e visual
       indica informação parcialmente duplicada.
    2. **Possíveis variáveis pouco relevantes:** `ash` é a candidata mais forte devido
       à grande sobreposição entre classes. `magnesium` também tem separação limitada,
       embora ainda possa contribuir em combinação com outras entradas.
    3. **Candidatas à remoção:** eu testaria retirar `total_phenols`, mantendo
       `flavanoids`, e avaliaria `ash` por validação cruzada. `magnesium` seria uma
       candidata secundária, não uma remoção automática.

    Essas decisões são hipóteses de AED, não uma seleção definitiva. O passo correto
    antes de otimizar um classificador é comparar, por validação cruzada, o desempenho
    com todas as entradas e com cada conjunto reduzido. Correlação não implica que uma
    variável seja inútil, e sobreposição univariada não elimina interações multivariadas.
    """),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3"}
client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbf.write(nb, NOTEBOOK)


# Imagens em alta resolução para o relatório.
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0,
            vmin=-1, vmax=1, fmt=".2f", annot_kws={"size": 7})
plt.title("Matriz de correlação - Wine")
plt.tight_layout()
plt.savefig(ASSETS / "matriz_correlacao.png", dpi=180)
plt.close()

g = sns.pairplot(df, hue="target", diag_kind="hist", corner=True,
                 plot_kws={"alpha": .65, "s": 16, "edgecolor": "none"},
                 diag_kws={"alpha": .6})
g.fig.suptitle("Matriz de scatterplots por classe", y=1.005)
g.savefig(ASSETS / "matriz_scatterplots.png", dpi=130)
plt.close("all")

for feature in features:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="target", y=feature, hue="target",
                palette="Set2", legend=False, ax=ax)
    ax.set_title(f"{feature} por classe")
    ax.set_xlabel("Classe (target)")
    plt.tight_layout()
    fig.savefig(BOXPLOTS / f"{feature.replace('/', '_')}.png", dpi=160)
    plt.close(fig)

fig, axes = plt.subplots(5, 3, figsize=(11, 15))
for ax, feature in zip(axes.flat, features):
    sns.boxplot(data=df, x="target", y=feature, hue="target", palette="Set2",
                legend=False, ax=ax, fliersize=3)
    ax.set_title(feature, fontsize=9)
    ax.set_xlabel("target", fontsize=8)
    ax.set_ylabel("")
for ax in axes.flat[len(features):]:
    ax.axis("off")
fig.suptitle("Boxplots individuais por classe", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, .98])
fig.savefig(ASSETS / "boxplots_resumo.png", dpi=170)
plt.close(fig)


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER,
                          textColor=colors.HexColor("#17365D"), spaceAfter=12))
styles.add(ParagraphStyle(name="H1Blue", parent=styles["Heading1"],
                          textColor=colors.HexColor("#17365D"), spaceBefore=10, spaceAfter=7))
styles.add(ParagraphStyle(name="BodyJust", parent=styles["BodyText"], alignment=TA_JUSTIFY,
                          leading=14, spaceAfter=7))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#A8B8C8"))
    canvas.line(2*cm, 1.35*cm, A4[0]-2*cm, 1.35*cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#4B5D6B"))
    canvas.drawString(2*cm, .95*cm, "Aprendizado de Máquina - PT-02 AED")
    canvas.drawRightString(A4[0]-2*cm, .95*cm, f"Página {doc.page}")
    canvas.restoreState()

doc = BaseDocTemplate(str(REPORT), pagesize=A4, rightMargin=1.8*cm, leftMargin=1.8*cm,
                      topMargin=1.7*cm, bottomMargin=1.7*cm,
                      title="PT-02 - Análise Exploratória de Dados", author="Valentim S. A. Neto")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=footer))

story = [Spacer(1, 2.2*cm), Paragraph("PT-02", styles["TitleCenter"]),
         Paragraph("Análise Exploratória do dataset Wine", styles["TitleCenter"]),
         Spacer(1, .8*cm), Paragraph("Valentim S. A. Neto", styles["Heading2"]),
         Paragraph("Matrícula: 2019027210", styles["BodyText"]),
         Paragraph("Disciplina: Aprendizado de Máquina", styles["BodyText"]),
         Paragraph("Professor: Frederico Coelho", styles["BodyText"]),
         Spacer(1, 5.5*cm), Paragraph("Relatório de análise exploratória", styles["BodyJust"]),
         PageBreak()]

def h(text): story.append(Paragraph(text, styles["H1Blue"]))
def p(text): story.append(Paragraph(text, styles["BodyJust"]))

h("1. Dataset e variáveis")
p("O Wine é um dataset clássico de classificação com 178 amostras, 13 variáveis numéricas positivas e três classes. Cada linha contém resultados de análises químicas de um vinho. Não há valores ausentes.")
p("As entradas são: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315_of_diluted_wines e proline. A saída é target, com as classes 0, 1 e 2, associadas a class_0, class_1 e class_2.")
p("Fonte consultada: documentação oficial do scikit-learn, load_wine (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html).")

h("2. Resumo estatístico")
summary = df[features].describe().loc[["mean", "std", "min", "max"]].T.round(2)
table_data = [["Variável", "Média", "Desvio", "Mín.", "Máx."]] + [[idx, *row] for idx, row in summary.iterrows()]
t = Table(table_data, colWidths=[6.8*cm, 2.1*cm, 2.1*cm, 2.1*cm, 2.1*cm], repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
                       ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                       ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                       ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
                       ("FONTSIZE", (0,0), (-1,-1), 7.5), ("ALIGN", (1,1), (-1,-1), "RIGHT"),
                       ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#B8C4CE")),
                       ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF3F7")]),
                       ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
story += [t, Spacer(1, .2*cm)]
p("As amplitudes são muito diferentes: proline varia de 278 a 1.680, enquanto outras medidas ficam próximas de zero. Em uma etapa de modelagem, algoritmos sensíveis à escala devem receber dados padronizados.")

story.append(PageBreak())
h("3. Matriz de correlações")
story.append(Image(str(ASSETS / "matriz_correlacao.png"), width=16.5*cm, height=12.4*cm))
p(f"Entre as variáveis de entrada, o único par que supera |r| = 0,80 é total_phenols e flavanoids (r = {cor.loc['total_phenols','flavanoids']:.3f}). A relação forte indica informação linear parcialmente duplicada. A correlação com target deve ser interpretada com cautela, pois os números 0, 1 e 2 são códigos de classes, não uma medida contínua.")

story.append(PageBreak())
h("4. Scatterplots")
story.append(Image(str(ASSETS / "matriz_scatterplots.png"), width=17.0*cm, height=17.0*cm))
p("Os scatterplots reforçam a proximidade entre total_phenols e flavanoids. Flavanoids, od280/od315_of_diluted_wines, proline, color_intensity e alcohol exibem separações úteis entre classes. Ash apresenta forte sobreposição, e magnesium também separa pouco quando observado isoladamente. Nenhuma variável separa perfeitamente as três classes.")

story.append(PageBreak())
h("5. Boxplots e outliers")
story.append(Image(str(ASSETS / "boxplots_resumo.png"), width=13.6*cm, height=18.5*cm))
p(f"Os boxplots foram construídos individualmente para respeitar as escalas. Pelo critério de Tukey aplicado globalmente, foram encontrados outliers em: {outlier_text}. Esses pontos não foram removidos: um outlier visual pode ser uma observação válida e a decisão de exclusão exigiria investigação adicional.")
p("Ash possui distribuições bastante sobrepostas nas três classes. Magnesium também apresenta grande sobreposição. Em contraste, flavanoids, od280/od315_of_diluted_wines, proline, color_intensity e alcohol exibem deslocamentos claros de mediana ou intervalo entre pelo menos duas classes.")

story.append(PageBreak())
h("6. Conclusões e respostas")
p(f"<b>1. Quais variáveis têm alta correlação?</b> Total_phenols e flavanoids, com r = {cor.loc['total_phenols','flavanoids']:.3f}. A proximidade de 1 e o padrão quase linear no scatterplot mostram que variam juntas; por isso, parte de sua informação é redundante.")
p("<b>2. Quais parecem não ajudar a distinguir as classes?</b> Ash é a principal candidata, pois suas caixas e pontos se sobrepõem amplamente. Magnesium também parece pouco discriminativa isoladamente. Isso não prova irrelevância multivariada: uma variável fraca sozinha ainda pode ajudar em interação com outras.")
p("<b>3. Quais variáveis considerar remover?</b> Eu testaria remover total_phenols e manter flavanoids, que preserva o principal sinal do par redundante. Também avaliaria retirar ash. Magnesium seria uma candidata secundária, apenas se testes de validação mostrarem que sua retirada não reduz o desempenho.")
p("A seleção proposta é uma hipótese baseada na AED. Para uma decisão definitiva, deve-se comparar por validação cruzada o modelo completo com versões reduzidas. Essa etapa evita transformar correlação ou sobreposição visual em uma conclusão causal ou automática.")

h("Referência")
p("SCIKIT-LEARN DEVELOPERS. load_wine: Load and return the wine dataset (classification). Documentação oficial. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html. Acesso em: 27 ago. 2026.")

doc.build(story)
print(json.dumps({"notebook": str(NOTEBOOK), "report": str(REPORT), "pairs": pairs,
                  "outliers": outliers}, ensure_ascii=False, indent=2))

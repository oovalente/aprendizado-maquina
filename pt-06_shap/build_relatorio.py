"""Relatório sem códigos, a partir dos resultados executados."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
DEST = OUT / 'pdf' / '2019027210_Valentim_S_A_Neto_Pratica6.pdf'

def gerar_relatorio():
    r = json.loads((OUT/'resultados.json').read_text(encoding='utf-8'))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Texto', fontSize=10.5, leading=15, spaceAfter=10))
    styles.add(ParagraphStyle(name='Titulo', fontSize=22, leading=27, textColor=colors.HexColor('#17365d'), spaceAfter=16))
    styles.add(ParagraphStyle(name='Secao', fontSize=15, leading=20, textColor=colors.HexColor('#17365d'), spaceAfter=12))
    story=[]
    def p(text, style='Texto'): story.append(Paragraph(text, styles[style]))
    def fig(name, height=340):
        im=Image(str(OUT/'figuras'/f'{name}.png'))
        scale=min(475/im.drawWidth, height/im.drawHeight)
        im.drawWidth*=scale; im.drawHeight*=scale
        story.extend([im, Spacer(1,10)])
    def table(rows, widths):
        t=Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17365d')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eff4fa'),colors.white])]))
        story.extend([t,Spacer(1,12)])
    def n(x, digits=3): return f'{x:.{digits}f}'.replace('.',',')
    def page(): story.append(PageBreak())
    p('Regressão logística e interpretação com SHAP','Titulo')
    p('<b>Prática 6 | Aprendizado de Máquina</b><br/>Valentim Soares de Andrade Neto<br/>Matrícula: 2019027210')
    p('1. Dados e análise exploratória','Secao')
    p('O Breast Cancer Wisconsin (Diagnostic), disponibilizado pelo scikit-learn, reúne 569 amostras de tumores: 212 malignos (classe 0) e 357 benignos (classe 1). Os 30 atributos numéricos descrevem núcleos celulares em imagens de punção aspirativa por agulha fina. Não foram encontrados valores ausentes ou registros com atributos duplicados.')
    p('São medidas dez características: raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal. Para cada uma são fornecidos a média, o erro padrão e a medida worst, correspondente à média dos três maiores valores. As unidades e amplitudes diferem, justificando a padronização.')
    fig('distribuicoes',300)
    p('Figura 1. Distribuições de seis atributos nas duas classes. Tumores malignos apresentam, em geral, maiores medidas de tamanho e irregularidade. A sobreposição impede uma regra perfeita baseada apenas em um atributo.')
    page()
    p('2. Correlações e preparação','Secao')
    fig('correlacoes',450)
    p('Figura 2. Correlações de Pearson entre os 30 atributos. Raio, perímetro e área têm forte associação, assim como várias medidas médias e worst. A redundância reduz a possibilidade de interpretar a importância individual como um efeito isolado.')
    p('Os dados foram divididos em 80% para treino (455) e 20% para teste (114), com semente 42, sem estratificação, seguindo o exemplo do enunciado. O treino contém 169 malignos e 286 benignos; o teste contém 43 malignos e 71 benignos. Não há imputação, pois não há dados ausentes.')
    p('O StandardScaler foi ajustado exclusivamente no treino. Cada atributo teve sua média de treino subtraída e foi dividido pelo respectivo desvio padrão. As mesmas transformações foram aplicadas ao teste. A exploração do conjunto completo tem finalidade descritiva, sem seleção de variáveis ou ajuste de hiperparâmetros.')
    page()
    p('3. Modelo e desempenho','Secao')
    p('Foi treinada uma regressão logística com regularização L2, C=1, solver lbfgs e limite de 2.000 iterações. Todos os 30 atributos foram usados. A classe 0 corresponde a maligno; sua probabilidade foi obtida diretamente da coluna correspondente do modelo. O limiar de classificação é 0,5.')
    table([['Métrica no teste','Resultado'],['Acurácia',n(r['acuracia']*100,2)+'%'],['AUC-ROC (maligno positivo)',n(r['auc_maligno'],6)],['Sensibilidade para maligno',n(r['sensibilidade_maligno']*100,2)+'%'],['Especificidade',n(r['especificidade']*100,2)+'%']],[320,155])
    fig('avaliacao',250)
    p('Figura 3. Matriz de confusão e curva ROC. Foram acertados 111 de 114 diagnósticos: 41 malignos e 70 benignos. Dois malignos foram classificados como benignos (falsos negativos) e um benigno como maligno (falso positivo).')
    p('A acurácia mede decisões no limiar adotado; a AUC mede a ordenação dos riscos ao variar o limiar. A AUC elevada não elimina erros na decisão nem comprova calibração das probabilidades. O teste foi usado apenas para avaliação final e interpretação.')
    page()
    p('4. Importância global com SHAP','Secao')
    p('O LinearExplainer usa os 455 registros de treino padronizados como referência, com masker Independent. A saída explicada é f(x) = log[P(benigno)/P(maligno)]. O valor base E[f(X)] é '+n(r['base'],6)+' e representa a média dos log-odds no treino. Não corresponde à probabilidade média.')
    p('A soma do valor base e das 30 contribuições produz f(x). A probabilidade benigna é 1/(1+exp(-f(x))); a maligna é seu complemento. Contribuições positivas favorecem benigno e negativas favorecem maligno. A reconstrução foi verificada em todas as 114 amostras; o maior erro numérico foi '+f"{r['erro_aditividade']:.2e}"+'.')
    fig('shap_bar',340)
    p('Figura 4. Média do valor absoluto das contribuições SHAP no teste, em unidades de log-odds. O ranking mede intensidade média, sem indicar por si só a direção do efeito.')
    p('Os três atributos mais influentes foram worst texture ('+n(r['ranking'][0]['media_abs_shap'])+'), mean concave points ('+n(r['ranking'][1]['media_abs_shap'])+') e radius error ('+n(r['ranking'][2]['media_abs_shap'])+'). Textura caracteriza variações na imagem; pontos côncavos descrevem irregularidade do contorno; radius error mede a variabilidade estimada do raio.')
    page()
    p('5. Direção e interpretação global','Secao')
    fig('shap_beeswarm',380)
    p('Figura 5. Cada ponto é uma amostra de teste. Vermelho indica valor alto do atributo padronizado, azul indica valor baixo; o eixo horizontal mostra a contribuição para o log-odds de benignidade.')
    p('Nos atributos de maior importância, valores altos tendem a produzir contribuições negativas, favorecendo malignidade. Isso é compatível com a expectativa de que alterações na textura, maior irregularidade e medidas elevadas de tamanho auxiliem a distinguir tumores malignos. Todos os dez primeiros atributos do ranking têm coeficiente negativo neste ajuste.')
    p('Para o explicador independente de um modelo linear, a contribuição é o coeficiente multiplicado pela diferença entre o valor padronizado do paciente e a média de referência. Assim, um atributo abaixo da média pode favorecer benignidade mesmo que seu coeficiente seja negativo.')
    p('A hipótese de referência independente não reparte o crédito entre variáveis correlacionadas. Raio, perímetro e área podem substituir parcialmente informações uns dos outros, e o ranking depende da regularização, da amostra e do modelo. As contribuições são associações preditivas, não efeitos causais nem incrementos diretos em pontos percentuais de probabilidade.')
    for i, patient in enumerate(r['pacientes'],1):
        page()
        p(f'6.{i}. Explicação do paciente {i}', 'Secao')
        p(f"Registro de índice {patient['indice']} (índice original iniciado em zero). Diagnóstico real: <b>{patient['real']}</b>. Classificação: <b>{patient['predito']}</b>. Probabilidade de malignidade: <b>{n(patient['p_maligno']*100,4)}%</b>. Os três casos são os primeiros registros de teste na ordem retornada pela divisão, sem seleção pelo resultado.")
        fig(f'paciente_{i}',330)
        p(f"Figura {5+i}. Waterfall em log-odds de benignidade. O valor base {n(patient['base'],4)} é ajustado pelas contribuições até f(x) = {n(patient['score'],4)}. As barras vermelhas favorecem benigno; as azuis favorecem maligno. Os atributos no gráfico estão padronizados; a tabela abaixo usa valores originais.")
        table([['Atributo','Valor original','SHAP']] + [[x['variavel'],n(x['valor_original'],5),n(x['shap'],4)] for x in patient['principais']],[245,115,115])
        desc='; '.join(x['variavel']+' favorece '+('benigno' if x['shap']>0 else 'maligno') for x in patient['principais'])
        p('As quatro maiores contribuições em magnitude: '+desc+'. A decisão considera a soma de todos os atributos, inclusive os agrupados na última barra do gráfico.')
    page()
    p('7. Discussão e conclusão','Secao')
    p('O modelo obteve 97,37% de acurácia e AUC de 0,997380 na divisão utilizada. Textura worst, média de pontos côncavos e erro padrão do raio foram os maiores destaques do SHAP. Os resultados são compatíveis com diferenças de textura, tamanho e irregularidade observadas entre as classes, embora as correlações impeçam atribuir importância exclusivamente a um fenômeno independente.')
    p('O paciente 1 apresentou contribuições de sentidos opostos e foi classificado como benigno, com risco estimado de malignidade de 11,36%. Nos pacientes 2 e 3, as maiores contribuições favoreceram malignidade, com probabilidades estimadas de 99,9991% e 99,6921%. Os três diagnósticos foram corretamente classificados.')
    p('A análise utiliza uma única divisão aleatória de uma base de 569 observações. Não foram avaliadas a estabilidade por validação cruzada, a calibração das probabilidades ou a generalização para outras populações. Os dois falsos negativos mostram que o desempenho agregado deve ser interpretado junto aos erros por classe. A atividade é um estudo didático de classificação e explicação de um modelo.')
    p('Reprodução e verificações','Secao')
    p('O notebook acompanha a atividade com execução completa, figuras, estatísticas descritivas, ranking, contribuições SHAP e predições exportadas. Foram verificadas a identidade aditiva do SHAP, a fórmula linear de cada contribuição e a correspondência entre log-odds e probabilidades. A padronização e a referência SHAP usam somente o treino.')
    p('Versões: '+', '.join(k+' '+v for k,v in r['versoes'].items())+'.')
    p('Referências','Secao')
    for label,url in [('scikit-learn: descrição do Breast Cancer Wisconsin','https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html'),('SHAP: LinearExplainer e hipóteses da referência','https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html'),('SHAP: gráfico waterfall','https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html')]:
        p(f'<link href="{url}" color="#17365d">{label}</link>')
    def footer(canvas, doc):
        canvas.setFont('Helvetica',9); canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(54,28,'Prática 6 | Valentim Soares de Andrade Neto'); canvas.drawRightString(A4[0]-54,28,str(doc.page))
    SimpleDocTemplate(str(DEST), pagesize=A4, leftMargin=54,rightMargin=54,topMargin=45,bottomMargin=48,title='Prática 6 - Regressão logística e SHAP',author='Valentim Soares de Andrade Neto').build(story,onFirstPage=footer,onLaterPages=footer)
    print(DEST)

if __name__ == '__main__': gerar_relatorio()

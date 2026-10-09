"""Produz o relatório sem códigos a partir dos resultados do notebook."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'
DEST=OUT/'pdf'/'2019027210_Valentim_S_A_Neto_Pratica7.pdf'

def gerar_relatorio():
    r=json.loads((OUT/'resultados.json').read_text(encoding='utf-8'))
    DEST.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Texto',fontSize=10,leading=14,spaceAfter=9))
    styles.add(ParagraphStyle(name='Titulo',fontSize=23,leading=28,textColor=colors.HexColor('#17365d'),spaceAfter=15))
    styles.add(ParagraphStyle(name='Secao',fontSize=14,leading=19,textColor=colors.HexColor('#17365d'),spaceAfter=12))
    story=[]
    def p(s,style='Texto'): story.append(Paragraph(s,styles[style]))
    def page(): story.append(PageBreak())
    def n(x,d=2): return f'{x:.{d}f}'.replace('.',',')
    def fig(name,h=220):
        im=Image(str(OUT/'figuras'/(name+'.png')))
        scale=min(487/im.drawWidth,h/im.drawHeight); im.drawWidth*=scale; im.drawHeight*=scale
        story.extend([im,Spacer(1,9)])
    def table(rows,widths):
        t=Table(rows,colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17365d')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eff4fa'),colors.white])]))
        story.extend([t,Spacer(1,12)])
    p('Dilema bias-variância','Titulo')
    p('<b>Prática 7 | Aprendizado de Máquina</b><br/>Valentim Soares de Andrade Neto<br/>Matrícula: 2019027210')
    p('1. Objetivo, dados e método','Secao')
    p('Investigar subajuste, ajuste adequado e sensibilidade à amostra em modelos de regressão linear com atributos polinomiais de graus 1, 2 e 10. A função geradora é f(x) = 10 + 2x + 3x², com x uniforme em [0,10] e y = f(x) + ruído gaussiano independente de média zero.')
    p('O texto do enunciado cita N(0,1), mas o código fornecido usa desvio padrão 10. Adotamos o código: cenário base com 100 dados e desvio padrão 10; ruído maior com 100 dados e desvio padrão 30; triplo de dados com 300 dados e desvio padrão 10. As variâncias do ruído são 100, 900 e 100, respectivamente.')
    table([['Cenário','Dados','Treino','Teste','Desvio do ruído'],['Base','100','80','20','10'],['Ruído maior','100','80','20','30'],['Triplo de dados','300','240','60','10']],[139,63,63,63,159])
    p('A geração principal utiliza RandomState(42), equivalente à semente do exemplo, e a divisão utiliza 80% para treino e 20% para teste, random_state=42. Mínimos quadrados ajustam os coeficientes somente no treino, com intercepto e sem regularização. Os graus foram definidos pelo enunciado; o teste serve para avaliação.')
    p('Para estabilidade numérica, aplicamos a transformação fixa z=(x-5)/5 antes de gerar as potências. Essa transformação preserva os polinômios possíveis em x e não aprende informações do teste. MSE é a média dos erros quadráticos; R² mede a redução do erro em relação à média do conjunto avaliado. MSE é expresso em unidades de y²; R² não tem unidade e pode ser negativo.')
    p('Além da divisão principal, executamos 200 repetições por cenário, com sementes 0 a 199. Elas ajudam a distinguir um resultado particular do comportamento da família de modelos, totalizando 1.800 ajustes adicionais.')
    page()
    p('2. Resultados da divisão principal','Secao')
    table([['Cenário','Grau','MSE treino','MSE teste','R² treino','R² teste']]+[[v['cenario'],str(v['grau']),n(v['mse_treino']),n(v['mse_teste']),n(v['r2_treino'],4),n(v['r2_teste'],4)] for v in r['metricas']],[127,40,80,80,80,80])
    fig('ajustes',205)
    p('Figura 1. Dados de treino e teste, função verdadeira e curvas ajustadas. A reta não acompanha a curvatura. Os polinômios de graus 2 e 10 se aproximam da relação geradora; as diferenças locais do grau 10 refletem sua flexibilidade.')
    p('No cenário base, o grau 1 apresenta MSE de treino 672,42 e teste 549,97, maiores que os dos outros modelos. O grau 2 apresenta o menor MSE de teste, 63,58. O grau 10 reduz o erro de treino para 77,04, mas aumenta ligeiramente o de teste para 66,12. Portanto, nesta divisão o grau 10 não apresenta um teste muito ruim, embora não melhore a generalização.')
    p('O R² da reta permanece alto porque a variação total de y é grande. Isso não elimina seu subajuste: o MSE e a curva revelam que a estrutura quadrática foi omitida. Valores de MSE abaixo da variância do ruído em uma amostra de teste pequena são possíveis por flutuação amostral; não significam eliminar o erro irredutível.')
    page()
    p('3. Estabilidade em 200 repetições','Secao')
    table([['Cenário','Grau','MSE treino\nmédio','MSE teste\nmédio','MSE teste\nmediano','Teste P90']]+[[v['cenario'],str(v['grau']),n(v['mse_treino_medio']),n(v['mse_teste_medio']),n(v['mse_teste_mediano']),n(v['mse_teste_p90'])] for v in r['resumo']],[127,40,80,80,80,80])
    fig('repeticoes',180)
    p('Figura 2. Distribuição do MSE de teste, em escala logarítmica. Caixas representam quartis; pontos extremos foram omitidos somente no desenho e permanecem nas médias e percentis. P90 é o valor abaixo do qual estão 90% dos resultados.')
    p('Na base, o MSE médio de teste é 624,12 para grau 1, 102,43 para grau 2 e 125,34 para grau 10. O grau 10 obtém menor erro médio de treino (86,74 versus 96,44), mas pior desempenho de teste. O grau 2 tem o melhor desempenho médio nos três cenários.')
    p('Os cenários de 100 dados compartilham x, divisão e ruído padronizado em cada semente; apenas sua amplitude muda. O cenário de 300 dados mantém o mesmo mecanismo gerador, mas não é uma ampliação aninhada da amostra anterior. As comparações de uma divisão podem refletir também a composição dos dados; as repetições reduzem a dependência de uma única realização.')
    page()
    p('4. Estimativa de bias e variância','Secao')
    p('Em uma grade fixa de 301 pontos em [0,10], calculamos a previsão média entre repetições. Bias² é a média do quadrado da diferença entre essa previsão e f(x); variância é a média da variância populacional das previsões. A decomposição empírica do erro contra f(x) em bias² + variância foi verificada numericamente. Contra uma nova resposta ruidosa, acrescentamos a variância do ruído.')
    table([['Cenário','Grau','Bias²','Variância','Ruído','MSE esperado\nna grade']]+[[v['cenario'],str(v['grau']),n(v['bias2']),n(v['variancia']),n(v['ruido']),n(v['mse_esperado_grade'])] for v in r['decomposicao']],[127,40,80,80,65,95])
    fig('estabilidade',180)
    p('Figura 3. Previsão média e faixa entre percentis 10 e 90 no cenário base. A faixa descreve a dispersão dos modelos ajustados; não é intervalo de confiança nem intervalo preditivo para y. A sensibilidade do grau 10 aparece principalmente nas bordas.')
    p('O grau 1 mantém bias² próximo de 507: mais dados não removem a curvatura omitida. Na base, a variância é 3,95 para grau 2 e 32,61 para grau 10. Com 300 dados, cai para 1,34 e 5,52. Com ruído maior, cresce para 35,55 e 293,48, respectivamente.')
    p('Bias² e variância são estimativas finitas de 200 repetições. A grade inclui as bordas e não tem exatamente a distribuição do teste aleatório; por isso seu MSE esperado não deve coincidir com o MSE médio de teste. O pequeno bias² dos polinômios não prova bias populacional diferente de zero.')
    page()
    p('5. Respostas à discussão','Secao')
    p('<b>1. Qual modelo teve erro alto em treino e teste (alto bias)?</b><br/>O grau 1. Na base, MSE 672,42 no treino e 549,97 no teste. Ele não contém x², e sua estimativa de bias² é 506,75 na grade. Trata-se de subajuste.')
    p('<b>2. Qual modelo teve baixo erro de treino e alto erro de teste (alta variância)?</b><br/>O grau 10 apresenta maior risco de sobreajuste. Na divisão 42 a diferença é pequena, sem erro de teste muito alto. Nas repetições da base, o treino médio cai de 96,44 (grau 2) para 86,74, mas o teste médio sobe de 102,43 para 125,34. Sua variância estimada é cerca de 8,3 vezes a do grau 2.')
    p('<b>3. Qual modelo tem o melhor desempenho geral?</b><br/>O grau 2, com menor MSE de teste nos três cenários na divisão principal e nas médias das repetições. Ele representa a função verdadeira com poucos parâmetros: intercepto e dois coeficientes, contra onze parâmetros no grau 10.')
    p('<b>4. Como a complexidade impacta bias e variância?</b><br/>Do grau 1 para 2, o bias cai fortemente porque a curvatura passa a ser representável. Do grau 2 para 10, a flexibilidade adicional pouco acrescenta à estrutura verdadeira e aumenta a variância. O erro de treino diminui, mas o erro de teste pode aumentar. Essa tendência depende da amostra e não é uma regra determinística para toda divisão.')
    p('<b>5. Por que o grau 10 pode ter teste ruim mesmo com treino bom?</b><br/>Termos adicionais capturam parte do ruído do treino. Quando os dados mudam, os coeficientes e as curvas variam mais, especialmente nas bordas. O treino bom mede adaptação àquela amostra, sem garantir generalização. A premissa de teste ruim não ocorreu de forma acentuada na divisão 42; as repetições mostram a maior sensibilidade.')
    p('<b>6. Aumentar o ruído altera os resultados? Por quê?</b><br/>Sim. De desvio 10 para 30, a variância do ruído cresce nove vezes. Na divisão principal, MSE de teste do grau 2 sobe de 63,58 para 572,26 e do grau 10 de 66,12 para 595,08, também nove vezes: nesses modelos a função verdadeira pertence ao espaço ajustado, e os mesmos resíduos padronizados foram usados. O erro irredutível e a instabilidade aumentam; o grau 1 mantém também seu erro de especificação.')
    p('<b>7. O que ocorre ao triplicar os dados?</b><br/>A variância dos estimadores tende a diminuir. O MSE médio de teste do grau 10 cai de 125,34 para 106,61; o do grau 2 fica próximo ao piso de ruído (102,43 para 102,22). O grau 1 continua com alto bias. Na divisão principal, o MSE de teste dos graus 2 e 10 aumenta ligeiramente: mais dados não garantem melhora em cada realização, e o novo conjunto de teste contém outros ruídos.')
    p('Reprodução e limites','Secao')
    p('Notebook executado, figuras e tabelas CSV acompanham o relatório. Verificações: valores finitos, 1.800 ajustes adicionais, MSE de treino não crescente entre graus e identidade da decomposição. Simulação unidimensional, 200 repetições e domínio [0,10]; não avalia extrapolação. Versões: '+', '.join(k+' '+v for k,v in r['versoes'].items())+'.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(54,27,'Prática 7 | Valentim Soares de Andrade Neto'); canvas.drawRightString(A4[0]-54,27,str(doc.page))
    SimpleDocTemplate(str(DEST),pagesize=A4,leftMargin=54,rightMargin=54,topMargin=42,bottomMargin=45,title='Prática 7 - Dilema bias-variância',author='Valentim Soares de Andrade Neto').build(story,onFirstPage=footer,onLaterPages=footer)
    print(DEST)

if __name__=='__main__': gerar_relatorio()

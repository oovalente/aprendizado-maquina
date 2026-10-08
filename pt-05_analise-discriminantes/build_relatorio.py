"""Monta um PDF sem códigos com os resultados exportados por build_pt05.py."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'output'
DESTINO = OUTPUT / 'pdf' / '2019027210_Valentim_S_A_Neto_Pratica5.pdf'


def gerar_relatorio():
    resultados = json.loads((OUTPUT / 'resultados.json').read_text(encoding='utf-8'))
    original = {r['Modelo']: r for r in resultados['original']}
    deslocado = {r['Modelo']: r for r in resultados['deslocado']}
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TituloPT', fontName='Helvetica-Bold', fontSize=21,
                              leading=25, textColor=colors.HexColor('#17365d'), spaceAfter=12))
    styles.add(ParagraphStyle(name='SecaoPT', fontName='Helvetica-Bold', fontSize=15,
                              leading=19, textColor=colors.HexColor('#17365d'), spaceAfter=10))
    styles.add(ParagraphStyle(name='TextoPT', fontName='Helvetica', fontSize=10.3,
                              leading=14, spaceAfter=9))
    styles.add(ParagraphStyle(name='LegendaPT', fontName='Helvetica', fontSize=9,
                              leading=12, textColor=colors.HexColor('#475569'), spaceAfter=8))
    story = []

    def p(text, style='TextoPT'):
        story.append(Paragraph(text, styles[style]))

    def figura(nome, largura=16.7*cm, altura_max=9*cm):
        im = Image(str(OUTPUT / 'figuras' / f'{nome}.png'))
        escala = min(largura / im.drawWidth, altura_max / im.drawHeight)
        im.drawHeight *= escala
        im.drawWidth *= escala
        story.append(im)

    def tabela(linhas, larguras):
        t = Table(linhas, colWidths=larguras, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17365d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#eff4fa'), colors.white]),
            ('LINEBELOW', (0, 0), (-1, -1), .4, colors.HexColor('#cbd5e1')),
        ]))
        story.extend([t, Spacer(1, 10)])

    def numero(valor, casas=2):
        return f'{valor:.{casas}f}'.replace('.', ',')

    def resumo(cenario):
        linhas = [['Modelo', 'Acurácia', 'AUC manual', 'TP', 'FP', 'TN', 'FN']]
        for nome, r in cenario.items():
            linhas.append([nome, numero(100*r['Acurácia'])+'%', numero(r['AUC manual'], 4),
                           r['TP'], r['FP'], r['TN'], r['FN']])
        tabela(linhas, [2*cm, 2.7*cm, 3*cm, 2*cm, 2*cm, 2*cm, 2*cm])

    p('Análise discriminante<br/>LDA, QDA e curvas ROC', 'TituloPT')
    p('<b>Prática 5 | Aprendizado de Máquina</b><br/>'
      'Valentim Soares de Andrade Neto | Matrícula: 2019027210')
    p('1. Geração dos dados e protocolo', 'SecaoPT')
    p('Foram geradas 900 observações bidimensionais com semente 42. A classe 1 '
      '(azul, rótulo 0) contém 300 amostras de uma normal centrada em (0, 0), '
      'com covariância [[1; 0,8], [0,8; 1]]. A classe 2 (vermelha, rótulo 1) '
      'reúne dois componentes de 300 amostras: média (-2; -2), covariância '
      '[[1; -0,6], [-0,6; 1]], e média (-2; 1,5), covariância [[1; 0,6], [0,6; 1]].')
    figura('original_dados', 15.5*cm)
    p('Figura 1. Dados originais e médias geradoras. A classe 2 é uma mistura de '
      'duas gaussianas. Há sobreposição, apesar do título “linearmente separáveis” '
      'presente no exemplo do enunciado.', 'LegendaPT')
    cont = resultados['contagens']
    tabela([['Grupo', 'Total', 'Treino', 'Teste']] +
           [[classe, cont['Total'][classe], cont['Treino'][classe], cont['Teste'][classe]]
            for classe in cont['Total']], [6.7*cm, 3*cm, 3*cm, 3*cm])
    p('Foi usada a divisão do enunciado: 70% para treino e 30% para teste, semente '
      '42 e sem estratificação. LDA e QDA foram ajustados somente no treino, '
      'usando o mesmo teste. As probabilidades a priori foram estimadas no treino. '
      'A classe positiva da ROC é a classe 2 (rótulo 1).')

    story.append(PageBreak())
    p('2. Cenário original: fronteiras e acurácia', 'SecaoPT')
    figura('original_fronteiras')
    p('Figura 2. Fronteiras no limiar discriminante zero, sobre as 270 amostras '
      'de teste. A reta representa LDA; o contorno verde tracejado representa QDA.', 'LegendaPT')
    resumo(original)
    p('TP e TN são acertos das classes positiva e negativa; FP são amostras da '
      'classe 1 classificadas como classe 2; FN são amostras da classe 2 '
      'classificadas como classe 1. “Precisão” é interpretada como <b>acurácia</b>, '
      'conforme a métrica do enunciado: (TP + TN) / total. Ela difere de TP / (TP + FP).')
    ganho = 100*(original['QDA']['Acurácia']-original['LDA']['Acurácia'])
    p(f'<b>QDA teve o melhor desempenho:</b> acurácia de '
      f'{numero(100*original["QDA"]["Acurácia"])}%, contra '
      f'{numero(100*original["LDA"]["Acurácia"])}% do LDA, vantagem de '
      f'{numero(ganho)} pontos percentuais. Foram '
      f'{original["QDA"]["FP"]+original["QDA"]["FN"]} erros do QDA e '
      f'{original["LDA"]["FP"]+original["LDA"]["FN"]} do LDA. '
      f'O classificador que sempre escolhe a classe 2 teria acurácia de '
      f'{numero(100*cont["Teste"]["Classe 2 (y=1)"]/270)}%.')
    p('A fronteira do QDA parece mais ajustada à geometria observada: permite '
      'curvatura e considera as diferentes covariâncias, acompanhando melhor a '
      'nuvem estreita e inclinada da classe 1. LDA impõe uma covariância comum '
      'e uma fronteira linear. A melhora visual é acompanhada por menor erro no teste.')
    p('Mesmo assim, QDA aproxima a classe 2 inteira por uma única gaussiana. '
      'Sua maior flexibilidade não representa exatamente os dois componentes '
      'geradores e não garante superioridade em qualquer conjunto de dados.')

    story.append(PageBreak())
    p('3. ROC manual e escolha entre os métodos', 'SecaoPT')
    p('Para cada modelo, obtivemos o score discriminante do teste e variamos '
      'manualmente o limiar: a previsão é positiva se score for maior ou igual '
      'ao limiar. Contamos TP, FP, TN e FN e calculamos <b>TPR = TP / (TP + FN)</b> '
      'e <b>FPR = FP / (FP + TN)</b>. Não foram usadas funções prontas de ROC ou AUC.')
    p('Percorremos todos os scores distintos em ordem decrescente, precedidos '
      'por infinito positivo. Assim, incluímos (0, 0) e (1, 1) e tratamos '
      'empates juntos. A AUC foi calculada pela soma das áreas dos trapézios '
      'entre pontos consecutivos, sem escolher um limiar no conjunto de teste.')
    figura('original_roc', 14*cm)
    p('Figura 3. Curvas no mesmo gráfico. Os círculos indicam o limiar padrão '
      'zero; a diagonal é a referência de ordenação aleatória.', 'LegendaPT')
    p(f'QDA apresenta maior área sob a curva: '
      f'{numero(original["QDA"]["AUC manual"], 4)}, contra '
      f'{numero(original["LDA"]["AUC manual"], 4)}. Isso indica melhor ordenação '
      'global dos positivos em relação aos negativos nesta amostra; não significa '
      'melhor desempenho em todo limiar nem mede calibração das probabilidades.')
    p('<b>Quando usar LDA:</b> covariâncias semelhantes e separação aproximadamente '
      'linear, sobretudo com poucas amostras ou muitas variáveis. Compartilhar '
      'a covariância reduz parâmetros e variância. Exemplo: classificação de '
      'sensores cujos grupos têm dispersões semelhantes.')
    p('<b>Quando usar QDA:</b> covariâncias diferentes e fronteiras curvas, com '
      'dados suficientes por classe. Exemplo: controle de qualidade com grupos '
      'de peças que apresentam correlações diferentes entre medidas. Com poucas '
      'amostras ou colinearidade, suas covariâncias podem se tornar instáveis '
      'ou singulares, exigindo regularização ou redução de dimensão.')
    p('Com d atributos e K classes, a covariância requer d(d+1)/2 parâmetros '
      'no LDA e K d(d+1)/2 no QDA. Ambos assumem uma gaussiana por classe; '
      'neste exercício, a mistura da classe 2 viola essa hipótese.', 'LegendaPT')

    story.append(PageBreak())
    p('4. Classe 1 deslocada para (3, 3)', 'SecaoPT')
    p('Somamos (3, 3) às amostras da classe 1, preservando seus ruídos e '
      'covariância, a classe 2 e os índices de treino/teste. Ambos os modelos '
      'foram treinados novamente. A única mudança é a média da classe 1.')
    figura('deslocado_dados', 15.2*cm)
    p('Figura 4. A classe 1 se afasta dos dois componentes da classe 2, '
      'reduzindo a sobreposição.', 'LegendaPT')
    figura('deslocado_fronteiras', 15.2*cm)
    p('Figura 5. Fronteiras dos modelos retreinados e amostras de teste. '
      'A separação linear já se torna muito eficaz, embora as covariâncias '
      'continuem diferentes.', 'LegendaPT')

    story.append(PageBreak())
    p('5. ROC após deslocamento e conclusões', 'SecaoPT')
    figura('deslocado_roc', 13.7*cm)
    p('Figura 6. ROC calculada novamente pelo procedimento manual. As duas '
      'curvas se aproximam do canto superior esquerdo.', 'LegendaPT')
    resumo(deslocado)
    p(f'LDA passou de {numero(100*original["LDA"]["Acurácia"])}% para '
      f'{numero(100*deslocado["LDA"]["Acurácia"])}% de acurácia; QDA passou de '
      f'{numero(100*original["QDA"]["Acurácia"])}% para '
      f'{numero(100*deslocado["QDA"]["Acurácia"])}%. A vantagem do QDA caiu de '
      f'{numero(ganho)} para '
      f'{numero(100*(deslocado["QDA"]["Acurácia"]-deslocado["LDA"]["Acurácia"]))} '
      'ponto percentual. O deslocamento, ao reduzir a sobreposição, beneficia '
      'ambos e diminui o ganho da fronteira quadrática.')
    p(f'As AUCs atingem {numero(deslocado["LDA"]["AUC manual"], 6)} (LDA) e '
      f'{numero(deslocado["QDA"]["AUC manual"], 6)} (QDA). '
      'AUC perfeita do QDA significa que todos os positivos recebem scores '
      'maiores que os negativos nesta amostra. Não implica acurácia perfeita '
      'no limiar zero: ainda há um falso positivo. A ordenação e a decisão '
      'em um limiar fixo medem aspectos distintos.')
    p('A classe 2 permanece bimodal e as gaussianas têm suporte ilimitado. '
      'Portanto, a separação observada não prova erro populacional zero. '
      'Os resultados se limitam a uma amostra sintética e uma divisão fixa; '
      'não demonstram superioridade estatística geral. A escolha entre LDA '
      'e QDA deve considerar geometria, quantidade de dados e validação fora do treino.')
    p('<b>Reprodutibilidade e verificação.</b> O notebook executado contém '
      'todas as etapas. Foram verificados os extremos e a monotonicidade da '
      'ROC, as contagens, exemplos de scores empatados e a equivalência entre '
      'a AUC por trapézios e por comparação de pares. Os limiares e contagens '
      'foram exportados em quatro arquivos CSV.', 'LegendaPT')
    p('<b>Fonte:</b> Aula 12 - Exercício Análise Discriminantes e ROC.pdf, '
      'enunciado disponibilizado na pasta da prática. Todos os gráficos e '
      'resultados deste relatório foram gerados pela resolução.', 'LegendaPT')

    def rodape(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(2*cm, 1.1*cm, 'PT-05 | Valentim Soares de Andrade Neto | 2019027210')
        canvas.drawRightString(A4[0]-2*cm, 1.1*cm, str(doc.page))
        canvas.restoreState()

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(DESTINO), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                      topMargin=1.7*cm, bottomMargin=1.8*cm,
                      title='PT-05 - Análise discriminante: LDA, QDA e ROC',
                      author='Valentim Soares de Andrade Neto').build(
                          story, onFirstPage=rodape, onLaterPages=rodape)
    print(f'Relatório gerado: {DESTINO}')


if __name__ == '__main__':
    gerar_relatorio()

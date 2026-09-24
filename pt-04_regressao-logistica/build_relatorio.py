"""Gera o relatório da PT-04 com os gráficos do notebook executado."""
from pathlib import Path
from io import BytesIO
import base64
import json
import math

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'output' / 'pdf'
OUTPUT.mkdir(parents=True, exist_ok=True)
DESTINO = OUTPUT / '2019027210_Valentim_S_A_Neto_Pratica4.pdf'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Texto', fontName='Helvetica', fontSize=10.5, leading=15, spaceAfter=9))
styles.add(ParagraphStyle(name='Titulo', fontName='Helvetica-Bold', fontSize=21, leading=26, textColor=colors.HexColor('#17365D'), spaceAfter=14))
styles.add(ParagraphStyle(name='Secao', fontName='Helvetica-Bold', fontSize=14, leading=19, textColor=colors.HexColor('#17365D'), spaceBefore=10, spaceAfter=10))
styles.add(ParagraphStyle(name='Legenda', fontName='Helvetica', fontSize=9, leading=12, textColor=colors.HexColor('#475569'), spaceAfter=10))
story = []


def p(text, style='Texto'):
    story.append(Paragraph(text, styles[style]))


def tabela(rows, widths):
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17365D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#EFF4FA'), colors.white]),
        ('LINEBELOW', (0, 0), (-1, -1), 0.4, colors.HexColor('#CBD5E1')),
    ]))
    story.extend([t, Spacer(1, 10)])


nb = json.loads((ROOT / 'regressao_logistica.ipynb').read_text(encoding='utf-8'))
graficos = [base64.b64decode(o['data']['image/png']) for c in nb['cells']
            for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]


def grafico(index, width):
    im = Image(BytesIO(graficos[index]))
    im.drawHeight *= width / im.drawWidth
    im.drawWidth = width
    story.append(im)


p('Regressão logística para<br/>concessão de crédito', 'Titulo')
p('<b>Prática 4 | Aprendizado de Máquina</b><br/>Universidade Federal de Minas Gerais<br/>'
  'Valentim Soares de Andrade Neto | Matrícula: 2019027210<br/>Professor: Frederico Gualberto Ferreira Coelho')
p('1. Objetivo e metodologia', 'Secao')
p('Aplicar um modelo de regressão logística já treinado para estimar a probabilidade de inadimplência, '
  'interpretar seus coeficientes e discutir o efeito do limiar de classificação. O exercício é fictício; '
  'não foram ajustados novos coeficientes nem utilizados dados externos.')
p('<b>Modelo fornecido:</b><br/>z = -4,5 + 0,08 × Idade - 0,00003 × Renda + 1,2 × Atrasos + 0,9 × Divida<br/>'
  'p = 1 / (1 + exp(-z))')
p('Idade é medida em anos; Renda é anual, em reais; Atrasos corresponde ao número de atrasos no último ano; '
  'Divida vale 0 para dívida baixa e 1 para dívida alta. O evento positivo (Y = 1) é a inadimplência.')
p('2. Parte 1: probabilidades dos clientes', 'Secao')
dados = [('A', 25, 40000, 0, 0), ('B', 40, 50000, 2, 1), ('C', 55, 120000, 1, 0), ('D', 30, 80000, 3, 1)]
rows = [['Cliente', 'Idade', 'Renda (R$)', 'Atrasos', 'Dívida', 'Logit', 'p (%)']]
for nome, idade, renda, atrasos, divida in dados:
    z = -4.5 + .08*idade - .00003*renda + 1.2*atrasos + .9*divida
    prob = 100/(1+math.exp(-z))
    rows.append([nome, idade, f'{renda:,}'.replace(',', '.'), atrasos, divida,
                 f'{z:.2f}'.replace('.', ','), f'{prob:.2f}'.replace('.', ',')])
tabela(rows, [1.8*cm, 1.7*cm, 3*cm, 2*cm, 1.8*cm, 2.2*cm, 2.5*cm])
p('Para A, por exemplo, z = -4,5 + 0,08 × 25 - 0,00003 × 40.000 = -3,7, resultando em '
  'p = 0,024127, ou 2,41%. B apresenta o maior risco estimado (62,25%). Para D, z = 0 e a probabilidade é 50%.')
p('A implementação usa uma sigmoide numericamente estável, baseada em logaddexp. Foram conferidos os '
  'quatro logits, as probabilidades, as classificações e o comportamento da função para valores extremos. '
  'Todas as verificações do notebook passaram.')
story.append(PageBreak())
p('3. Parte 2: interpretação dos coeficientes', 'Secao')
p('<b>a) Influência no aumento do risco.</b> Por incremento de uma unidade, Atrasos tem o maior coeficiente '
  'positivo (1,2), seguido de Divida (0,9) e Idade (0,08). Um atraso adicional aumenta mais o logit que um '
  'ano adicional de idade ou a mudança de dívida baixa para alta, mantendo as outras variáveis constantes. '
  'Isso não define importância global: as variáveis têm unidades e amplitudes diferentes. Por exemplo, '
  '15 anos acrescentam 1,2 ao logit, o mesmo que um atraso.')
p('<b>b) Coeficiente negativo da renda.</b> Mantidas as demais características, maior renda reduz o logit '
  'e a probabilidade prevista. Essa associação é compatível com maior capacidade de pagamento, mas não '
  'demonstra causalidade. R$ 10.000 adicionais de renda anual reduzem o logit em 0,3.')
p('<b>c) Efeito de um atraso.</b> Um atraso adicional aumenta em 1,2 o logit, isto é, o logaritmo das '
  'chances de inadimplência. Não representa aumento de 1,2 na probabilidade nem de 120 pontos percentuais: '
  'a mudança na probabilidade depende do risco inicial.')
p('<b>d) Razões de chances.</b> As chances são p/(1-p). A razão de chances associada a um incremento '
  'unitário é exp(coeficiente), mantendo as demais variáveis constantes.')
tabela([['Variação', 'Coeficiente', 'Razão de chances'], ['Atrasos: +1', '1,2', '3,3201'],
        ['Dívida: baixa para alta', '0,9', '2,4596']], [7*cm, 4*cm, 4*cm])
p('Um atraso adicional multiplica as chances por aproximadamente 3,32, um aumento de 232%. '
  'Passar de dívida baixa para alta multiplica as chances por 2,46, um aumento de 146%. '
  'Esses multiplicadores se aplicam às chances, não diretamente às probabilidades.')
p('4. Parte 3: classificações por limiar', 'Secao')
p('<b>a-b)</b> Nesta parte são usadas as probabilidades fornecidas para E-I. A regra é prever Y = 1 '
  'quando p é maior ou igual ao limiar; caso contrário, prever Y = 0.')
tabela([['Cliente', 'Probabilidade', 'Limiar 0,5', 'Limiar 0,7'],
        ['E', '15%', '0', '0'], ['F', '42%', '0', '0'], ['G', '58%', '1', '0'],
        ['H', '72%', '1', '1'], ['I', '89%', '1', '1']], [3*cm, 4*cm, 4*cm, 4*cm])
p('0 = adimplente previsto; 1 = inadimplente previsto. Apenas G muda de classe entre os dois limiares.', 'Legenda')
story.append(PageBreak())
p('4. Parte 3: decisão econômica (continuação)', 'Secao')
grafico(0, 15*cm)
p('Figura 1. Probabilidades fornecidas para E-I e os dois limiares de decisão.', 'Legenda')
p('<b>c) Banco.</b> Entre 0,5 e 0,7, escolheria 0,5 sob a hipótese de que conceder crédito a um '
  'inadimplente custa mais que perder um bom cliente. O limiar menor sinaliza mais casos de risco e '
  'reduz falsos negativos, embora possa aumentar falsos positivos. Um falso negativo é um inadimplente '
  'classificado como adimplente; um falso positivo é um adimplente classificado como inadimplente.')
p('<b>d) Campanha de marketing.</b> Escolheria 0,7 caso o objetivo fosse ampliar o público, houvesse '
  'análise de crédito posterior e a receita esperada justificasse maior tolerância ao risco. G passaria '
  'a ser elegível, enquanto H e I permaneceriam sinalizados. Se a oferta pré-aprovada implicasse '
  'concessão efetiva sem nova análise, manteria 0,5 sob a hipótese conservadora do item anterior.')
p('Portanto, não existe um limiar ideal apenas por se tratar de banco ou marketing. A escolha depende '
  'da margem, valor da operação, perda esperada e custo de oportunidade. Em um modelo simplificado, '
  'com custos constantes, custo zero para acertos e probabilidades calibradas, o limiar ótimo é '
  '<b>C<sub>FP</sub> / (C<sub>FP</sub> + C<sub>FN</sub>)</b>. Quanto maior o custo do falso negativo, '
  'menor o limiar indicado.')
p('O enunciado não fornece custos nem resultados reais de inadimplência. Assim, as escolhas acima são '
  'condicionais: não é possível demonstrar qual limiar maximiza o lucro ou calcular métricas observadas '
  'de classificação.')
story.append(PageBreak())
p('5. Parte 4: análise crítica', 'Secao')
p('<b>Afirmação 1: probabilidade de 58% significa inadimplência certa.</b> A afirmação confunde '
  'probabilidade com certeza. O resultado individual de G é desconhecido. Se o modelo estiver bem '
  'calibrado, entre muitos casos com previsão próxima de 58%, espera-se aproximadamente 58% de '
  'inadimplência. Uma classificação positiva não elimina essa incerteza.')
p('<b>Afirmação 2: pessoas mais velhas sempre têm maior risco.</b> O efeito positivo da idade vale '
  'mantendo renda, atrasos e dívida constantes. Na comparação entre pessoas diferentes, as outras '
  'características podem compensá-lo. C, com 55 anos, tem risco previsto menor que B, com 40 anos. '
  'Além disso, o coeficiente não comprova causalidade nem garante validade fora da população estudada.')
p('<b>Afirmação 3: elevar o limiar para 0,9 melhora o modelo.</b> Alterar o limiar muda as decisões, '
  'não os coeficientes nem as probabilidades. O conjunto de positivos previstos diminui: falsos '
  'positivos não aumentam, falsos negativos não diminuem e a revocação não aumenta, quando há '
  'positivos reais. A precisão pode aumentar, mas isso não é garantido em uma amostra finita.')
p('Com limiar 0,9, todos os clientes E-I são classificados como adimplentes. Não há falsos positivos '
  'nesse grupo, mas qualquer inadimplente real seria falso negativo. A precisão TP/(TP+FP) fica '
  'indefinida porque não há previsões positivas. A revocação TP/(TP+FN) é zero se houver inadimplentes '
  'reais e indefinida se não houver. Sem os rótulos reais, não se podem determinar TP, FP, FN e TN.')
p('6. Bônus: inclinação e sensibilidade', 'Secao')
p('<b>1) Curva mais inclinada.</b> O modelo z = -1,2 + 5X tem transição mais íngreme que '
  'z = -1,2 + 0,5X. A derivada é dp/dX = b × p(1-p), com máximo b/4: 1,25 para b = 5 '
  'e 0,125 para b = 0,5. Os centros das transições são X = 0,24 e X = 2,4, respectivamente.')
p('<b>2) Sensibilidade.</b> O coeficiente 5 produz maior sensibilidade perto de sua região de '
  'transição, mas não necessariamente em todo X. Na saturação, perto de p = 0 ou p = 1, '
  'a derivada se aproxima de zero; nesse caso, o modelo com coeficiente 0,5 pode ser mais sensível '
  'no mesmo valor de X.')
p('<b>3) Coeficientes excessivamente grandes.</b> Podem amplificar ruído perto da fronteira, gerar '
  'previsões extremas e indicar sobreajuste ou separação completa/quase completa. A escala da variável '
  'precisa ser considerada. Padronização, regularização e validação ajudam a investigar esses casos. '
  'Uma implementação ingênua da exponencial também pode apresentar overflow.')
story.append(PageBreak())
p('6. Bônus: comparação gráfica (continuação)', 'Secao')
grafico(1, 16*cm)
p('Figura 2. Curvas sigmoides e derivadas para os dois coeficientes do bônus.', 'Legenda')
p('7. Conclusão', 'Secao')
p('A aplicação do modelo resultou em probabilidades de 2,41%, 62,25%, 7,59% e 50% para A, B, C e D, '
  'respectivamente. Atrasos apresenta o maior efeito positivo por unidade sobre o logit, mas a '
  'comparação de importância exige considerar as escalas das variáveis. Razões de chances devem '
  'ser diferenciadas de mudanças na probabilidade.')
p('A comparação entre os limiares 0,5 e 0,7 alterou apenas a decisão sobre G. Um limiar maior não '
  'implica um modelo melhor: a escolha deve considerar os custos dos erros e os objetivos da '
  'aplicação. Como não existem rótulos reais no exercício, as conclusões se limitam às previsões '
  'e às propriedades matemáticas do modelo, sem alegar desempenho empírico.')
p('8. Fontes e reprodução', 'Secao')
p('Enunciado: <i>Exercicio_Regressao_Logistica.pdf</i>, disponível na pasta da prática 4.<br/>'
  'Implementação e verificações: <i>regressao_logistica.ipynb</i>.<br/>'
  'Geração do notebook: <i>build_pt04.py</i>.<br/>'
  'Geração deste relatório: <i>build_relatorio.py</i>.')
p('Os gráficos foram extraídos das saídas do notebook executado. As probabilidades da Parte 1 '
  'foram recalculadas a partir dos coeficientes fornecidos. Para reproduzir, instale as dependências '
  'de requirements.txt, execute python build_pt04.py e, depois, python build_relatorio.py.')


def rodape(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#64748B'))
    canvas.drawString(2*cm, 1.2*cm, 'Valentim S. A. Neto | Prática 4 - Regressão logística')
    canvas.drawRightString(A4[0]-2*cm, 1.2*cm, f'{doc.page}')


doc = SimpleDocTemplate(str(DESTINO), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                        topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title='Prática 4 - Regressão logística', author='Valentim Soares de Andrade Neto')
doc.build(story, onFirstPage=rodape, onLaterPages=rodape)
print(DESTINO)

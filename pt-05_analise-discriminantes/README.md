# PT-05 - Análise discriminante

Resolução completa de `Aula 12 - Exercício Análise Discriminantes e ROC.pdf`.

- `analise_discriminantes.ipynb`: notebook executado com as três partes e o experimento com centro em (3, 3).
- `output/pdf/2019027210_Valentim_S_A_Neto_Pratica5.pdf`: relatório sem códigos, para entrega.
- `output/figuras/`: seis gráficos dos dois cenários.
- `output/*_roc_*.csv`: limiares, TP, FP, TN, FN, TPR e FPR das quatro curvas manuais.
- `output/resultados.json`: métricas, contagens da divisão e versões utilizadas.

## Reprodução

Nesta pasta, com Python 3.10 ou superior:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python build_pt05.py
python build_relatorio.py
```

Também é possível executar todas as células pelo Jupyter ou VS Code com o
ambiente selecionado, usando esta pasta como diretório de trabalho. Gere o
notebook antes do relatório: este consome os resultados e gráficos exportados.
Não há downloads de dados. Os geradores sobrescrevem os artefatos da prática.

## Convenções e resultados

Classe 1 (azul) recebe rótulo 0; classe 2 (vermelha) recebe rótulo 1 e é o
evento positivo da ROC. “Precisão” no enunciado corresponde à **acurácia**.
A geração usa a sequência aleatória original (`RandomState(42)`), n=300, e a
divisão 70%/30% sem estratificação. O deslocamento preserva os mesmos ruídos
e índices de treino/teste, retreinando ambos os modelos.

| Centro da classe 1 | Acurácia LDA | Acurácia QDA | AUC LDA | AUC QDA |
|---|---:|---:|---:|---:|
| (0, 0) | 84,44% | 94,07% | 0,913214 | 0,972747 |
| (3, 3) | 99,26% | 99,63% | 0,999549 | 1,000000 |

A ROC percorre todos os scores distintos, além de infinito positivo, com
contagens manuais e AUC por soma de trapézios. Não utiliza funções prontas
de ROC/AUC. O notebook verifica exemplos conhecidos, empates, extremos,
monotonicidade e equivalência da área com a comparação de pares. A classe 2
é uma mistura de gaussianas: as hipóteses de LDA e QDA não são exatas aqui.

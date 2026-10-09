# PT-07 - Dilema bias-variância

Resolução completa do exercício, com autoria e matrícula conforme as práticas anteriores.

- `bias_variancia.ipynb`: notebook executado, regressões de graus 1, 2 e 10, gráficos, MSE e R².
- `output/pdf/2019027210_Valentim_S_A_Neto_Pratica7.pdf`: relatório sem códigos, com respostas às sete perguntas.
- `output/figuras/`: ajustes, distribuição do erro nas repetições e estabilidade das previsões.
- `output/`: dados sintéticos, métricas dos três cenários, 200 repetições por cenário, decomposição de bias² e variância e versões.

## Reprodução

Python 3.12 ou superior, nesta pasta:

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python build_pt07.py
python build_relatorio.py
```

O notebook também pode ser executado no Jupyter, com esta pasta como diretório de trabalho. Os geradores sobrescrevem seus artefatos. Não há download de dados.

## Decisões e resultados

O texto do enunciado cita ruído N(0,1), mas seu código usa desvio padrão 10. Seguimos o código. Cenários: n=100 e σ=10; n=100 e σ=30; n=300 e σ=10. Divisão 80/20 com semente 42. Regressão sem regularização, usando potências de z=(x-5)/5 para estabilidade numérica, sem mudar o espaço de modelos.

O grau 2 obteve o menor MSE de teste em todos os cenários na divisão principal e na média de 200 repetições. Na base, os MSE de teste são 549,97 (grau 1), 63,58 (grau 2) e 66,12 (grau 10). O grau 10 não teve teste muito ruim nesta divisão; sua maior variância aparece nas repetições. Aumentar a amostra reduz sua variância estimada na grade de 32,61 para 5,52. A grade e o teste têm distribuições distintas; as métricas não devem coincidir exatamente.

As verificações no notebook confirmam valores finitos, 1.800 ajustes adicionais, erro de treino não crescente com o grau e a identidade empírica bias² + variância. A faixa gráfica de percentis representa dispersão dos modelos, não um intervalo preditivo.

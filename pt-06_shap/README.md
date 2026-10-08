# PT-06 - Regressão logística e SHAP

Resolução de `Aula 14 - Exercício SHAP.pdf`.

- `shap_regressao_logistica.ipynb`: notebook executado com exploração, modelo, avaliação e SHAP global e local de três pacientes.
- `output/pdf/2019027210_Valentim_S_A_Neto_Pratica6.pdf`: relatório sem códigos para entrega.
- `output/figuras/`: distribuições, correlações, avaliação, summary bar, beeswarm e três waterfalls.
- `output/`: estatísticas, ranking, contribuições SHAP, atributos originais do teste, predições e resultados/versões.

## Reprodução

Python 3.12 ou superior, nesta pasta:

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python build_pt06.py
python build_relatorio.py
```

O notebook também pode ser executado pelo Jupyter, com esta pasta como diretório de trabalho. Os geradores sobrescrevem seus artefatos. O dataset já acompanha o scikit-learn; não há download de dados.

## Convenções e resultados

Divisão 80/20 com semente 42, sem estratificação, como no exemplo. StandardScaler ajustado somente no treino; regressão logística L2, C=1. Acurácia: **97,37%**; AUC com maligno positivo: **0,997380**. Dois falsos negativos e um falso positivo em 114 amostras de teste.

Mantidos os rótulos 0=maligno, 1=benigno. SHAP explica **log-odds de benignidade**: positivo favorece benigno, negativo favorece maligno. A referência independente contém todos os 455 registros de treino. Aditividade, fórmula linear e conversão para probabilidades são verificadas no notebook. As explicações não são causais e dependem da referência; atributos correlacionados exigem cuidado na interpretação.

Os três primeiros registros de teste são analisados, índices originais 204, 70 e 131. O relatório contém fontes oficiais e discussão das limitações.

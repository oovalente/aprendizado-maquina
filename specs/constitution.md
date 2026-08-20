# Constituição do Projeto

## 1. Missão

Este projeto existe para ensinar, de forma prática e progressiva, dois fundamentos:

* programação em Python aplicada à análise de dados;
* compreensão intuitiva e matemática da Lei dos Grandes Números.

O projeto deve permitir que uma pessoa sem conhecimento prévio sobre a Lei dos Grandes Números consiga observar seu comportamento por meio de um experimento computacional simples.

Ao mesmo tempo, deve apresentar uma primeira aproximação da lógica utilizada em problemas de aprendizado de máquina, especialmente a representação de um resultado por meio de uma variável-alvo binária.

A prioridade das decisões do projeto será:

[
\boxed{
\text{Aprendizado}

>

\text{Fidelidade ao enunciado}

>

\text{Simplicidade}

>

\text{Qualidade técnica}

>

\text{Extensibilidade}
}
]

---

# 2. Público-alvo

O material deve ser compreensível principalmente para:

* estudantes que nunca estudaram a Lei dos Grandes Números;
* estudantes iniciando programação em Python;
* pessoas interessadas em compreender a lógica básica utilizada em aprendizado de máquina.

Não deve ser assumido conhecimento avançado de estatística, Python ou Machine Learning.

Sempre que possível, conceitos devem ser introduzidos inicialmente por meio de intuição e somente depois formalizados matematicamente.

---

# 3. Produto

O projeto possuirá dois artefatos principais.

## 3.1 Notebook didático

Será desenvolvido um **Jupyter Notebook** responsável por:

* gerar os dados;
* explicar o experimento;
* executar os cálculos;
* produzir os resultados;
* gerar visualizações;
* apresentar interpretações intermediárias.

O notebook será tratado como uma **aula executável**.

Sua estrutura deverá seguir preferencialmente:

[
\boxed{\text{Pergunta}}
\rightarrow
\boxed{\text{Explicação}}
\rightarrow
\boxed{\text{Código}}
\rightarrow
\boxed{\text{Resultado}}
\rightarrow
\boxed{\text{Interpretação}}
]

O código não deve apenas funcionar. Deve ser possível compreender por que cada etapa existe.

---

## 3.2 Relatório final

O produto entregue será um relatório contendo:

* proporção de alunos aprovados no conjunto completo;
* proporção encontrada nas amostras;
* comparação entre os resultados;
* gráfico;
* interpretação relacionada à Lei dos Grandes Números.

O relatório final não deverá conter código, conforme solicitado pelo exercício.

O notebook será a ferramenta utilizada para gerar as evidências apresentadas no relatório.

---

# 4. Escopo

O programa deverá cumprir exatamente o experimento solicitado.

Serão utilizadas amostras de tamanho:

[
n \in {10,50,100,500}
]

Não faz parte do escopo principal criar experimentos adicionais com outros tamanhos de amostra.

Cada tamanho terá exatamente uma amostra aleatória no resultado oficial.

---

# 5. Dados

Como o exercício não fornece um conjunto de dados, será criado um **dataset sintético** especificamente para fins didáticos.

Essa característica deverá ser explicitamente informada no notebook e no relatório quando pertinente.

O conjunto terá:

[
N=1000
]

estudantes.

Cada registro deverá possuir as variáveis previstas pelo exercício:

* horas de estudo por semana;
* número de faltas;
* nota final;
* idade.

---

# 6. Geração dos dados

Os dados não deverão ser completamente independentes e arbitrários.

A nota final deverá possuir uma relação plausível com algumas das características do estudante.

Conceitualmente:

[
\text{nota final}
=================

\text{nota base}
+
\alpha(\text{horas de estudo})
------------------------------

\beta(\text{faltas})
+
\epsilon
]

onde:

* (\alpha) representa o efeito positivo das horas de estudo;
* (\beta) representa o efeito negativo das faltas;
* (\epsilon) representa variabilidade individual e fatores não modelados.

A nota deverá respeitar:

[
0 \leq \text{nota final} \leq 100
]

O objetivo dessa relação não é construir um modelo fiel da realidade educacional, mas criar dados plausíveis que facilitem a interpretação do exercício.

---

# 7. Idade

A idade será gerada apenas como uma característica dos estudantes.

Ela **não deverá influenciar a nota final**.

Isso evita introduzir uma relação artificial que não contribui para o objetivo principal do experimento.

Uma faixa plausível deverá ser utilizada, por exemplo:

[
17 \leq idade \leq 30
]

---

# 8. Regra de aprovação

A variável-alvo será:

```text
aprovado
```

e deverá seguir a regra:

[
aprovado =
\begin{cases}
1, & \text{se nota final} \geq 60\
0, & \text{se nota final} < 60
\end{cases}
]

Essa regra é determinística.

Não será treinado um algoritmo de Machine Learning para decidir se o aluno foi aprovado.

Embora o exercício utilize o termo **Aprendizado Supervisionado**, esta prática seguirá literalmente as atividades propostas no enunciado e trabalhará apenas com a criação da variável-alvo binária.

Algoritmos como:

* regressão logística;
* árvore de decisão;
* Random Forest;
* redes neurais;

ficam explicitamente fora do escopo.

---

# 9. Interpretação estatística

A variável `aprovado` deverá ser interpretada como uma variável aleatória binária:

[
X_i =
\begin{cases}
1, & \text{aluno aprovado}\
0, & \text{aluno reprovado}
\end{cases}
]

Sua média é:

[
\bar X_n =
\frac{X_1+X_2+\cdots+X_n}{n}
]

Como os valores são apenas (0) e (1), essa média corresponde exatamente à proporção de alunos aprovados:

[
\hat p_n = \bar X_n
]

Essa relação será o principal elo entre o experimento computacional e a Lei dos Grandes Números.

---

# 10. Lei dos Grandes Números

A explicação deverá seguir a regra:

[
\boxed{\text{Intuição primeiro, matemática depois}}
]

Inicialmente, deverá ser explicado que amostras pequenas estão mais sujeitas à influência do acaso.

Conforme o número de observações aumenta, variações individuais possuem menor influência sobre o resultado agregado.

Depois dessa interpretação intuitiva, será apresentada a relação:

[
\hat p_n \rightarrow p
]

quando:

[
n \rightarrow \infty
]

onde:

* (\hat p_n) é a proporção observada na amostra;
* (p) é a proporção verdadeira da população.

O notebook poderá também introduzir a relação:

[
\operatorname{Var}(\bar X)
==========================

\frac{\sigma^2}{n}
]

para mostrar por que a variabilidade da média diminui conforme o tamanho da amostra aumenta.

---

# 11. Não exigir convergência monotônica

O programa não deverá manipular os resultados para fazer com que cada amostra maior seja necessariamente mais próxima da proporção populacional.

Por exemplo, é perfeitamente possível encontrar:

[
|\hat p_{100}-p|

>

|\hat p_{50}-p|
]

em uma determinada execução.

Isso não viola a Lei dos Grandes Números.

A lei descreve um comportamento estatístico à medida que o tamanho da amostra aumenta, e não uma obrigação de melhoria em cada amostra individual.

Caso esse comportamento apareça nos resultados, ele deverá ser explicado em vez de corrigido ou escondido.

---

# 12. Reprodutibilidade

Todos os processos aleatórios utilizados para:

* gerar o dataset;
* retirar as amostras;

deverão utilizar sementes fixas.

Por exemplo:

```python
random_state = 42
```

O objetivo é garantir que duas execuções completas do notebook produzam os mesmos resultados.

Os valores exatos das sementes poderão ser definidos durante a implementação, mas a reprodutibilidade é um requisito obrigatório.

---

# 13. Bibliotecas

O princípio será:

[
\boxed{\text{utilizar o mínimo de bibliotecas necessárias}}
]

Bibliotecas principais:

### pandas

Responsável por:

* armazenamento tabular;
* manipulação dos dados;
* criação da coluna `aprovado`;
* amostragem;
* cálculo das proporções.

### matplotlib

Responsável pelas visualizações.

### numpy

Poderá ser utilizado quando facilitar a geração dos dados aleatórios ou operações numéricas.

Seu uso não é obrigatório quando uma solução simples puder ser implementada com `pandas`.

Não será utilizado `scikit-learn`, pois não haverá treinamento de modelos.

---

# 14. Arquitetura

O projeto não deverá possuir arquitetura complexa.

Não são necessários:

* classes;
* múltiplos módulos;
* design patterns;
* frameworks;
* pipelines sofisticados;
* abstrações prematuras.

A arquitetura será essencialmente linear e didática:

[
\text{Gerar dados}
\rightarrow
\text{Inspecionar dados}
\rightarrow
\text{Criar alvo}
\rightarrow
\text{Calcular população}
\rightarrow
\text{Amostrar}
\rightarrow
\text{Comparar}
\rightarrow
\text{Visualizar}
\rightarrow
\text{Interpretar}
]

A simplicidade estrutural faz parte do objetivo educacional.

---

# 15. Validação

Não serão utilizados testes automatizados externos como `pytest`.

O próprio notebook deverá conter verificações simples para detectar erros importantes.

Exemplos:

```python
assert len(df) >= 500
```

```python
assert set(df["aprovado"].unique()).issubset({0, 1})
```

```python
assert df["nota_final"].between(0, 100).all()
```

As validações devem servir simultaneamente para aumentar a confiança no experimento e ensinar boas práticas de programação.

---

# 16. Visualização

O gráfico principal deverá possuir:

### Eixo X

Tamanho da amostra:

[
10,;50,;100,;500
]

### Eixo Y

Proporção de alunos aprovados.

Os quatro resultados deverão aparecer como pontos ou uma curva ligando os pontos.

Também deverá existir uma linha horizontal representando a proporção encontrada no conjunto completo:

[
p
]

Essa linha funcionará como referência visual.

O objetivo principal do gráfico será permitir observar se:

[
\hat p_n
]

se aproxima de:

[
p
]

à medida que:

[
n
]

cresce.

---

# 17. Estilo da implementação

Sempre que houver conflito entre código compacto e código didático, deverá ser preferido o código mais fácil de compreender.

Devem ser favorecidos:

* nomes de variáveis descritivos;
* operações explícitas;
* células pequenas;
* comentários úteis;
* explicações antes de conceitos novos.

Devem ser evitados:

* one-liners difíceis de interpretar;
* abstrações sem necessidade;
* otimizações prematuras;
* código excessivamente condensado.

---

# 18. Papel do agente

O agente será responsável pela implementação e explicação do projeto.

Seu papel não será apenas gerar código correto.

Ao implementar cada etapa relevante, deverá explicar:

1. o que será feito;
2. por que aquela etapa é necessária;
3. como o código funciona;
4. o que o resultado significa;
5. como ele se conecta ao objetivo geral do experimento.

O projeto deverá privilegiar compreensão, mesmo quando existir uma maneira mais rápida de produzir o mesmo resultado.

---

# 19. Definition of Done

O projeto será considerado concluído somente quando:

1. o Jupyter Notebook executar do início ao fim sem erros;

2. o dataset sintético possuir 1000 estudantes e for reproduzível;

3. estiver explicitamente informado que os dados são sintéticos;

4. todas as quatro variáveis do exercício estiverem presentes;

5. a coluna `aprovado` estiver corretamente criada;

6. a proporção total de alunos aprovados estiver calculada;

7. existirem amostras de:

[
10,\quad50,\quad100,\quad500
]

registros;

8. cada amostra possuir sua proporção de aprovados calculada;

9. os resultados estiverem organizados de forma comparável;

10. existir um gráfico mostrando a proporção em função do tamanho da amostra;

11. o gráfico possuir uma referência para a proporção do conjunto completo;

12. o notebook explicar intuitivamente a Lei dos Grandes Números;

13. o notebook apresentar posteriormente sua interpretação matemática;

14. existirem validações básicas utilizando `assert`;

15. o relatório final puder ser produzido apenas a partir das evidências geradas pelo notebook;

16. o relatório final não contiver código.

---

# 20. Regra para futuras decisões

Sempre que surgir uma nova funcionalidade, biblioteca, abstração ou alteração de escopo, deverá ser feita a pergunta:

> **Essa decisão contribui diretamente para aprender programação ou compreender a Lei dos Grandes Números?**

Se a resposta for não, a mudança deverá possuir outra justificativa suficientemente forte para entrar no projeto.

A Constituição deverá impedir que sofisticação técnica desnecessária retire o foco do objetivo educacional.

---

# 21. Princípio central

> **O projeto deve transformar a Lei dos Grandes Números de uma definição abstrata em um fenômeno que o estudante consegue gerar, observar, medir, visualizar e explicar com Python.**

Toda decisão futura de especificação e implementação deverá ser compatível com esse princípio.

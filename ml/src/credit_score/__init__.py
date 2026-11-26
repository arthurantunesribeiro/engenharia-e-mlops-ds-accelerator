"""Score de inadimplência de crédito: código compartilhado entre treino, batch, serving e monitoramento.

O mesmo pacote valida e prepara o dado em todos os caminhos, e vai empacotado dentro do modelo
registrado, o que evita o training-serving skew.

Módulos:
    features: entradas brutas → features do modelo.
    contract: contrato de entrada (regras de domínio).
    model: treino, métricas, score com faixa e motivos, e o wrapper MLflow.
    service: montagem da requisição do serviço de crédito.
    monitoring: classificação do drift do data profiling e alertas.
"""

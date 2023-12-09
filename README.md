# Projeto Classificador de Gênero de Filmes MLP

## Visão Geral
Este projeto tem como objetivo desenvolver um classificador de aprendizado de máquina capaz de categorizar filmes em gêneros com base em seu conteúdo visual. Ele utiliza o poder dos Perceptrons de Múltiplas Camadas (MLPs) para analisar e classificar filmes, extraindo e analisando características de quadros.

## Estrutura do Projeto

### Diretórios
- `data/`: Contém filmes e quadros.
  - `frames/`: Quadros extraídos para cada filme, organizados por nome de filme.
  - `movies/`: Arquivos de filmes em formatos como .mkv, .mp4, etc.
- `extraction_tools/`: Scripts para extração e pré-processamento de quadros.
- `models/`: Contém scripts de treinamento do modelo MLP e utilitários relacionados.
- `outputs/`: Armazenamento para modelos treinados e saídas.
- `src/`: Código-fonte do projeto.

## Instruções de Uso

1. Coloque os filmes em `data/movies/`. Cada filme deve estar no formato {gênero}_{nome}.{extensão}
2. Execute `frame_extractor.py` para extrair quadros.
3. Use `frame_preprocessor.py` para pré-processar os quadros.
4. Use `audio_extractor.py` para extrair os audios dos filmes.
5. Use `audio_features_refactor.py` para extrair features de audio.
6. Use `frequency_features_refactor.py` para extrair features de frequencia.
7. Use `frame_feature_extractor.py` para extrair features de video
8. Agregue características usando `feature_input_creator.py`.
9. Finalize a criacao do dataset com `csv_creator.npy`
10. Treine o MLP com `MLP_trainer.py`
11. Teste hiperparametros com `hyperparameter_optimizer.py`
12. Visualize características com `feature_visualization.py`.

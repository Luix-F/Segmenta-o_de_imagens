# Segmentação
Segmentação de imagens refere-se ao processo de dividir uma imagem digital em regiões ou objetos, com o objetivo de simplificar e/ou alterar sua representação para facilitar a análise. Geralmente, o resultado da segmentação de imagens é um conjunto de regiões ou objetos cujos pixels apresentam características semelhantes, como cor, intensidade, textura ou continuidade. Regiões adjacentes devem exibir diferenças significativas em relação à mesma característica.

## Artigo


Artigo "Implementação e Análise Comparativa de Técnicas de Segmentação de Imagens Baseadas em Grafos": 
https://pt.overleaf.com/read/ngpfydfywtmk#ff5714

## Compilação e execução

Utilize "esc" para fechar as janelas de visualização das imagens

Para compilar o código, certifique-se de ter o OpenCV4 instalado e execute o seguinte comando no terminal:

```bash
g++ NomeDoCodigo.cpp -o a `pkg-config --cflags --libs opencv4`
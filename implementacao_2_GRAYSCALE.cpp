#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>
#include <limits>
#include <iostream>

using namespace cv;
using namespace std;

const int INF = numeric_limits<int>::max();
const int dx[] = {-1, 1, 0, 0}; // 4 vizinhos: cima, baixo, esquerda, direita
const int dy[] = {0, 0, -1, 1};

struct Pixel {
    int y, x;
    int cost;

    // Para a fila de prioridade (min-heap)
    bool operator>(const Pixel& other) const {
        return cost > other.cost;
    }
};

// Semente: posição (x, y) e rótulo (label)
struct Seed {
    int x, y, label;
};

void iftSegmentation(const Mat& img, const vector<Seed>& seeds, Mat& labelsOut) {
    int height = img.rows;
    int width = img.cols;

    // Inicializa matrizes de custo e rótulo
    Mat cost(height, width, CV_32S, Scalar(INF));
    labelsOut = Mat(height, width, CV_8U, Scalar(0)); // saída

    priority_queue<Pixel, vector<Pixel>, greater<Pixel>> heap;

    // Inicializa sementes
    for (const Seed& seed : seeds) {
        cost.at<int>(seed.y, seed.x) = 0;
        labelsOut.at<uchar>(seed.y, seed.x) = seed.label;
        heap.push({seed.y, seed.x, 0});
    }

    // Propagação do caminho mínimo
    while (!heap.empty()) {
        Pixel p = heap.top();
        heap.pop();

        for (int i = 0; i < 4; ++i) {
            int ny = p.y + dy[i];
            int nx = p.x + dx[i];

            if (ny < 0 || ny >= height || nx < 0 || nx >= width)
                continue;

            int diff = abs(img.at<uchar>(p.y, p.x) - img.at<uchar>(ny, nx));
            int newCost = max(p.cost, diff); // custo do caminho mínimo

            if (newCost < cost.at<int>(ny, nx)) {
                cost.at<int>(ny, nx) = newCost;
                labelsOut.at<uchar>(ny, nx) = labelsOut.at<uchar>(p.y, p.x);
                heap.push({ny, nx, newCost});
            }
        }
    }
}
/*
int main() {
    // 1. Carregar imagem em tons de cinza
    Mat img = imread("cores.png", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Erro ao carregar imagem!" << endl;
        return -1;
    }

    // 2. Definir sementes (x, y, label)
    vector<Seed> seeds = {
        {133, 222, 100},  // região 1 (label 100)
        {299, 306, 200}, // região 2 (label 200)
        {140, 372, 300} // região 2 (label 200)
    };

    // 3. Rodar segmentação
    Mat segmented;
    iftSegmentation(img, seeds, segmented);

    // 4. Mostrar e salvar resultado
    imshow("Original", img);
    imshow("Segmentada", segmented);
    imwrite("resultado_segmentado.png", segmented);
    waitKey(0);
    return 0;
}*/


vector<Seed> seeds;
int currentLabel = 100;

void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        seeds.push_back({x, y, currentLabel});
        cout << "Semente adicionada: (" << x << ", " << y << ") label=" << currentLabel << endl;
        currentLabel += 100; // próximo clique será outra região
    }
}

int main() {
    Mat img = imread("./input/fl.png", IMREAD_GRAYSCALE);
    if (img.empty()) return -1;

    namedWindow("Clique para sementes");
    setMouseCallback("Clique para sementes", onMouse, nullptr);

    cout << "Clique em diferentes regiões para definir as sementes.\n";
    cout << "Pressione qualquer tecla quando terminar...\n";

    imshow("Clique para sementes", img);
    waitKey(0);  // espera os cliques

    // Continua com iftSegmentation normalmente:
    Mat segmented;
    iftSegmentation(img, seeds, segmented);
    imshow("Segmentada", segmented);
    imwrite("./output/resultado_segmentado_grayscale.png", segmented);
    waitKey(0);
}
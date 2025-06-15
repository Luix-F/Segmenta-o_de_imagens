#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// Estrutura para representar uma aresta
struct Edge {
    float weight;
    int u, v;
    Edge(float w, int u_, int v_) : weight(w), u(u_), v(v_) {}
    bool operator<(const Edge& other) const { return weight < other.weight; }
};

// Classe Disjoint-Set para gerenciar componentes
class DisjointSet {
public:
    vector<int> parent, rank, size;
    vector<float> int_val; // Diferença interna (Int(C))

    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        size.resize(n, 1);
        int_val.resize(n, 0.0f);
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]); // Compressão de caminho
        return parent[x];
    }

    void union_sets(int x, int y, float weight) {
        int px = find(x), py = find(y);
        if (px == py) return;
        // União por rank
        if (rank[px] < rank[py]) swap(px, py);
        parent[py] = px;
        size[px] += size[py];
        int_val[px] = max({int_val[px], int_val[py], weight});
        if (rank[px] == rank[py]) rank[px]++;
    }
};

// Gera arestas para um grafo 8-conectado
vector<Edge> get_edges(const Mat& image, float sigma) {
    Mat smoothed;
    if (sigma > 0) {
        GaussianBlur(image, smoothed, Size(0, 0), sigma);
    } else {
        smoothed = image.clone();
    }

    int height = image.rows, width = image.cols;
    vector<Edge> edges;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            // Direita
            if (x + 1 < width) {
                float weight = abs(smoothed.at<float>(y, x) - smoothed.at<float>(y, x + 1));
                edges.emplace_back(weight, idx, idx + 1);
            }
            // Baixo
            if (y + 1 < height) {
                float weight = abs(smoothed.at<float>(y, x) - smoothed.at<float>(y + 1, x));
                edges.emplace_back(weight, idx, idx + width);
            }
            // Diagonal: baixo-direita
            if (x + 1 < width && y + 1 < height) {
                float weight = abs(smoothed.at<float>(y, x) - smoothed.at<float>(y + 1, x + 1));
                edges.emplace_back(weight, idx, idx + width + 1);
            }
            // Diagonal: baixo-esquerda
            if (x > 0 && y + 1 < height) {
                float weight = abs(smoothed.at<float>(y, x) - smoothed.at<float>(y + 1, x - 1));
                edges.emplace_back(weight, idx, idx + width - 1);
            }
        }
    }
    return edges;
}

// Segmenta a imagem
Mat segment_image(const Mat& image, float sigma = 0.8f, float k = 300.0f, int min_size = 50) {
    int height = image.rows, width = image.cols;
    int n_vertices = height * width;

    // Converte a imagem para float
    Mat image_float;
    image.convertTo(image_float, CV_32F);

    // Inicializa Disjoint-Set
    DisjointSet ds(n_vertices);

    // Obtém arestas
    vector<Edge> edges = get_edges(image_float, sigma);

    // Ordena arestas por peso
    sort(edges.begin(), edges.end());

    // Processa arestas
    for (const auto& edge : edges) {
        int pu = ds.find(edge.u), pv = ds.find(edge.v);
        if (pu != pv) {
            float mint = min(ds.int_val[pu] + k / ds.size[pu], ds.int_val[pv] + k / ds.size[pv]);
            if (edge.weight <= mint) {
                ds.union_sets(edge.u, edge.v, edge.weight);
            }
        }
    }

    // Impõe tamanho mínimo
    for (const auto& edge : edges) {
        int pu = ds.find(edge.u), pv = ds.find(edge.v);
        if (pu != pv && (ds.size[pu] < min_size || ds.size[pv] < min_size)) {
            ds.union_sets(edge.u, edge.v, edge.weight);
        }
    }

    // Cria mapa de segmentação
    Mat segmentation(height, width, CV_32S);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            segmentation.at<int>(y, x) = ds.find(idx);
        }
    }

    // Re-rotula para inteiros consecutivos
    vector<int> labels;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            labels.push_back(segmentation.at<int>(y, x));
        }
    }
    sort(labels.begin(), labels.end());
    labels.erase(unique(labels.begin(), labels.end()), labels.end());
    unordered_map<int, int> label_map;
    for (size_t i = 0; i < labels.size(); ++i) {
        label_map[labels[i]] = i;
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            segmentation.at<int>(y, x) = label_map[segmentation.at<int>(y, x)];
        }
    }

    return segmentation;
}

int main() {
    // Carrega a imagem em escala de cinza
    Mat image = imread("./img/carFiltro.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Erro ao carregar a imagem!" << endl;
        return -1;
    }

    // Parâmetros
    float sigma = 0.8f;
    float k = 300.0f;
    int min_size = 50;

    // Segmenta a imagem
    Mat segmentation = segment_image(image, sigma, k, min_size);

    // Normaliza a segmentação para visualização
    double max_val;
    minMaxLoc(segmentation, nullptr, &max_val);
    Mat seg_vis;
    segmentation.convertTo(seg_vis, CV_8U, 255.0 / max_val);

    // Salva a imagem segmentada
    imwrite("./output-png/segmented_image_GRAYSCALE.jpg", seg_vis);

    return 0;
}
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <random>

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
vector<Edge> get_edges(const Mat& channel, float sigma) {
    Mat smoothed;
    if (sigma > 0) {
        GaussianBlur(channel, smoothed, Size(0, 0), sigma);
    } else {
        smoothed = channel.clone();
    }

    int height = channel.rows, width = channel.cols;
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

// Segmenta um canal da imagem
Mat segment_channel(const Mat& channel, float sigma, float k, int min_size) {
    int height = channel.rows, width = channel.cols;
    int n_vertices = height * width;

    // Converte o canal para float
    Mat channel_float;
    channel.convertTo(channel_float, CV_32F);

    // Inicializa Disjoint-Set
    DisjointSet ds(n_vertices);

    // Obtém arestas
    vector<Edge> edges = get_edges(channel_float, sigma);

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

    return segmentation;
}

// Intersecta as segmentações dos canais R, G, B
Mat intersect_segmentations(const Mat& seg_r, const Mat& seg_g, const Mat& seg_b) {
    int height = seg_r.rows, width = seg_r.cols;
    Mat segmentation(height, width, CV_32S);
    unordered_map<string, int> label_map;
    int next_label = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Cria uma chave única para o triplet (R, G, B)
            string key = to_string(seg_r.at<int>(y, x)) + "_" +
                         to_string(seg_g.at<int>(y, x)) + "_" +
                         to_string(seg_b.at<int>(y, x));
            if (label_map.find(key) == label_map.end()) {
                label_map[key] = next_label++;
            }
            segmentation.at<int>(y, x) = label_map[key];
        }
    }

    return segmentation;
}

// Gera uma cor RGB aleatória
Vec3b random_color(mt19937& rng) {
    uniform_int_distribution<int> dist(0, 255);
    return Vec3b(dist(rng), dist(rng), dist(rng));
}

// Segmenta uma imagem colorida e gera saída colorida
Mat segment_image(const Mat& image, float sigma = 0.8f, float k = 300.0f, int min_size = 50) {
    // Separa os canais R, G, B
    vector<Mat> channels;
    split(image, channels); // channels[0] = B, channels[1] = G, channels[2] = R

    // Segmenta cada canal
    Mat seg_r = segment_channel(channels[2], sigma, k, min_size); // R
    Mat seg_g = segment_channel(channels[1], sigma, k, min_size); // G
    Mat seg_b = segment_channel(channels[0], sigma, k, min_size); // B

    // Intersecta as segmentações
    Mat segmentation = intersect_segmentations(seg_r, seg_g, seg_b);

    // Re-rotula para inteiros consecutivos
    vector<int> labels;
    for (int y = 0; y < segmentation.rows; ++y) {
        for (int x = 0; x < segmentation.cols; ++x) {
            labels.push_back(segmentation.at<int>(y, x));
        }
    }
    sort(labels.begin(), labels.end());
    labels.erase(unique(labels.begin(), labels.end()), labels.end());

    // Gera cores aleatórias para cada rótulo
    mt19937 rng(random_device{}());
    unordered_map<int, Vec3b> color_map;
    for (int label : labels) {
        color_map[label] = random_color(rng);
    }

    // Cria imagem colorida de segmentação
    Mat seg_vis(segmentation.rows, segmentation.cols, CV_8UC3);
    for (int y = 0; y < segmentation.rows; ++y) {
        for (int x = 0; x < segmentation.cols; ++x) {
            int label = segmentation.at<int>(y, x);
            seg_vis.at<Vec3b>(y, x) = color_map[label];
        }
    }

    return seg_vis;
}

int main() {
    // Carrega a imagem colorida
    Mat image = imread("./img/carFiltro.png", IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Erro ao carregar a imagem!" << endl;
        return -1;
    }

    // Parâmetros
    float sigma = 0.8f;
    float k = 300.0f;
    int min_size = 50;

    // Segmenta a imagem
    Mat seg_vis = segment_image(image, sigma, k, min_size);

    // Salva a imagem segmentada
    imwrite("./output-png/segmented_image_COLOR.jpg", seg_vis);

    return 0;
}

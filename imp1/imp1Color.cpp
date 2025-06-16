#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>

using namespace cv;
using namespace std;

// ---------- Estrutura de Aresta ----------
struct Edge {
    float w;
    int u, v;
    bool operator<(const Edge& o) const { return w < o.w; }
};

// ---------- Disjoint-Set (Union-Find) ----------
class DSU {
public:
    vector<int> parent, size;
    vector<float> internal;

    DSU(int n) : parent(n), size(n, 1), internal(n, 0.0f) {
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    }

    void join(int x, int y, float w) {
        x = find(x); y = find(y);
        if (x == y) return;
        if (size[x] < size[y]) swap(x, y);
        parent[y] = x;
        size[x] += size[y];
        internal[x] = max({internal[x], internal[y], w});
    }
};

// ---------- Função hash para tupla 3D ----------
struct TupleHash {
    size_t operator()(const tuple<int, int, int>& t) const {
        auto [a, b, c] = t;
        return ((hash<int>()(a) ^ (hash<int>()(b) << 1)) >> 1) ^ (hash<int>()(c) << 1);
    }
};

// ---------- Construção do grafo ----------
vector<Edge> build_edges(const Mat& img) {
    int h = img.rows, w = img.cols;
    vector<Edge> edges;
    auto at = [&](int y, int x) { return img.at<float>(y, x); };

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            if (x + 1 < w) edges.push_back({abs(at(y, x) - at(y, x+1)), idx, idx+1});
            if (y + 1 < h) edges.push_back({abs(at(y, x) - at(y+1, x)), idx, idx+w});
            if (x + 1 < w && y + 1 < h)
                edges.push_back({abs(at(y, x) - at(y+1, x+1)), idx, idx+w+1});
            if (x > 0 && y + 1 < h)
                edges.push_back({abs(at(y, x) - at(y+1, x-1)), idx, idx+w-1});
        }
    return edges;
}

// ---------- Segmentação de um canal ----------
Mat segment_channel(const Mat& channel, float sigma, float k, int min_size) {
    Mat img;
    channel.convertTo(img, CV_32F);
    if (sigma > 0) GaussianBlur(img, img, Size(), sigma);

    int h = img.rows, w = img.cols, n = h * w;
    DSU dsu(n);
    auto edges = build_edges(img);
    sort(edges.begin(), edges.end());

    for (const auto& e : edges) {
        int u = dsu.find(e.u), v = dsu.find(e.v);
        if (u != v) {
            float thresh = min(dsu.internal[u] + k / dsu.size[u],
                               dsu.internal[v] + k / dsu.size[v]);
            if (e.w <= thresh)
                dsu.join(e.u, e.v, e.w);
        }
    }

    for (const auto& e : edges) {
        int u = dsu.find(e.u), v = dsu.find(e.v);
        if (u != v && (dsu.size[u] < min_size || dsu.size[v] < min_size))
            dsu.join(e.u, e.v, e.w);
    }

    Mat result(h, w, CV_32S);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            result.at<int>(y, x) = dsu.find(y * w + x);
    return result;
}

// ---------- Interseção entre segmentações ----------
Mat intersect_segmentations(const Mat& r, const Mat& g, const Mat& b) {
    int h = r.rows, w = r.cols;
    Mat result(h, w, CV_32S);
    unordered_map<tuple<int, int, int>, int, TupleHash> label_map;
    int next_label = 0;

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            auto key = make_tuple(
                r.at<int>(y, x),
                g.at<int>(y, x),
                b.at<int>(y, x)
            );
            if (label_map.count(key) == 0)
                label_map[key] = next_label++;
            result.at<int>(y, x) = label_map[key];
        }
    return result;
}

// ---------- Gera cor RGB aleatória ----------
Vec3b random_color(mt19937& rng) {
    uniform_int_distribution<int> dist(0, 255);
    return Vec3b(dist(rng), dist(rng), dist(rng));
}

// ---------- Pipeline completo de segmentação ----------
Mat segment_image(const Mat& image, float sigma = 0.8f, float k = 300.0f, int min_size = 50) {
    vector<Mat> channels(3);
    split(image, channels); // BGR

    Mat seg_r = segment_channel(channels[2], sigma, k, min_size);
    Mat seg_g = segment_channel(channels[1], sigma, k, min_size);
    Mat seg_b = segment_channel(channels[0], sigma, k, min_size);

    Mat labels = intersect_segmentations(seg_r, seg_g, seg_b);

    unordered_set<int> unique_labels;
    for (int y = 0; y < labels.rows; ++y)
        for (int x = 0; x < labels.cols; ++x)
            unique_labels.insert(labels.at<int>(y, x));

    mt19937 rng(random_device{}());
    unordered_map<int, Vec3b> color_map;
    for (int label : unique_labels)
        color_map[label] = random_color(rng);

    Mat output(labels.rows, labels.cols, CV_8UC3);
    for (int y = 0; y < labels.rows; ++y)
        for (int x = 0; x < labels.cols; ++x)
            output.at<Vec3b>(y, x) = color_map[labels.at<int>(y, x)];

    return output;
}

// ---------- Função principal ----------
int main() {
    Mat image = imread("./img/igFiltro.png");
    if (image.empty()) {
        cerr << "Erro ao carregar a imagem!" << endl;
        return -1;
    }

    float sigma = 0.8f;
    float k = 300.0f;
    int min_size = 50;

    Mat segmented = segment_image(image, sigma, k, min_size);    

    imwrite("./output-png/segmented_image_COLOR.jpg", segmented);
    return 0;
}
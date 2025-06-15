#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>
#include <limits>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

const int INF = numeric_limits<int>::max();
const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

struct Pixel {
    int y, x;
    int cost;
    bool operator>(const Pixel& other) const {
        return cost > other.cost;
    }
};

struct Seed {
    int x, y, label;
};

int colorCost(const Vec3b& a, const Vec3b& b) {
    int db = a[0] - b[0];
    int dg = a[1] - b[1];
    int dr = a[2] - b[2];
    return db * db + dg * dg + dr * dr;
}

void iftSegmentation(const Mat& img, const vector<Seed>& seeds, Mat& labelsOut) {
    int height = img.rows;
    int width = img.cols;

    Mat cost(height, width, CV_32S, Scalar(INF));
    labelsOut = Mat(height, width, CV_32S, Scalar(0));

    priority_queue<Pixel, vector<Pixel>, greater<Pixel>> heap;

    for (const Seed& seed : seeds) {
        cost.at<int>(seed.y, seed.x) = 0;
        labelsOut.at<int>(seed.y, seed.x) = seed.label;
        heap.push({seed.y, seed.x, 0});
    }

    while (!heap.empty()) {
        Pixel p = heap.top();
        heap.pop();

        for (int i = 0; i < 4; ++i) {
            int ny = p.y + dy[i];
            int nx = p.x + dx[i];

            if (ny < 0 || ny >= height || nx < 0 || nx >= width)
                continue;

            Vec3b colorP = img.at<Vec3b>(p.y, p.x);
            Vec3b colorN = img.at<Vec3b>(ny, nx);
            int diff = colorCost(colorP, colorN);
            int newCost = max(p.cost, diff);

            if (newCost < cost.at<int>(ny, nx)) {
                cost.at<int>(ny, nx) = newCost;
                labelsOut.at<int>(ny, nx) = labelsOut.at<int>(p.y, p.x);
                heap.push({ny, nx, newCost});
            }
        }
    }
}

vector<Seed> seeds;
int currentLabel = 100;

void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        seeds.push_back({x, y, currentLabel});
        cout << "Semente adicionada: (" << x << ", " << y << ") label=" << currentLabel << endl;
        currentLabel += 100;
    }
}

int main() {
    Mat img = imread("./input/fl.png", IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Erro ao carregar imagem!" << endl;
        return -1;
    }

    namedWindow("Clique para sementes");
    setMouseCallback("Clique para sementes", onMouse, nullptr);

    cout << "Clique em diferentes regiÃµes da imagem colorida para definir sementes.\n";
    cout << "Pressione qualquer tecla quando terminar...\n";

    imshow("Clique para sementes", img);
    waitKey(0);

    Mat segmented;
    iftSegmentation(img, seeds, segmented);

    Mat colorResult(img.size(), CV_8UC3);
    map<int, Vec3b> labelColors;
    RNG rng(12345);
    for (int y = 0; y < segmented.rows; ++y) {
        for (int x = 0; x < segmented.cols; ++x) {
            int lbl = segmented.at<int>(y, x);
            if (!labelColors.count(lbl))
                labelColors[lbl] = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            colorResult.at<Vec3b>(y, x) = labelColors[lbl];
        }
    }

    imshow("Segmentada", colorResult);
    imwrite("./output/resultado_segmentado_colorido.png", colorResult);
    waitKey(0);
    return 0;
}
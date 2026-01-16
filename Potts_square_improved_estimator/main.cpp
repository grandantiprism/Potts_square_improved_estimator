#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <string>
#include <chrono>
#include <queue>
#include <iomanip>
#include <algorithm>

using namespace std;

// ==========================================
// シミュレーションパラメータの設定
// ==========================================
struct Config {
    const int L = 128;               // 格子の一辺の長さ
    const int q = 4;                // Potts状態数 (q=3)
    
    const double beta_min = 1.095;
    const double beta_max = 1.105;
    const int beta_num = 20;
    
    const int n_warmup = 5000;
    const int n_measure = 1000000;
    
    const uint32_t seed = 12345;
};

class PottsSquare {
private:
    int L, N, q_states;
    vector<int> spins;
    mt19937 engine;
    uniform_real_distribution<double> dist_real;
    uniform_int_distribution<int> dist_site;
    uniform_int_distribution<int> dist_q_minus_1;

public:
    PottsSquare(int L, int q, uint32_t seed) :
        L(L), N(L * L), q_states(q), spins(L * L, 0),
        engine(seed), dist_real(0.0, 1.0),
        dist_site(0, L * L - 1), dist_q_minus_1(1, q - 1) {}

    inline int get_idx(int x, int y) {
        return ((x + L) % L) * L + ((y + L) % L);
    }

    int wolff_step(double beta) {
        int start_node = dist_site(engine);
        int old_spin = spins[start_node];
        
        // Potts模型の更新: 現在の値以外の(q-1)個の状態からランダムに選択
        int shift = dist_q_minus_1(engine);
        int new_spin = (old_spin + shift) % q_states;

        double p_add = 1.0 - exp(-beta);
        queue<int> que;

        spins[start_node] = new_spin;
        que.push(start_node);
        int cluster_size = 1;

        // 正方格子の4近傍 (x,y) に対して
        // (1,0), (-1,0), (0,1), (0,-1)
        int dx[] = {1, -1, 0, 0};
        int dy[] = {0, 0, 1, -1};

        while (!que.empty()) {
            int curr = que.front();
            que.pop();

            int cx = curr / L;
            int cy = curr % L;

            for (int i = 0; i < 4; ++i) {
                int neighbor = get_idx(cx + dx[i], cy + dy[i]);
                if (spins[neighbor] == old_spin) {
                    if (dist_real(engine) < p_add) {
                        spins[neighbor] = new_spin;
                        que.push(neighbor);
                        cluster_size++;
                    }
                }
            }
        }
        return cluster_size;
    }

    // Potts磁化の計算: m = (q*max_rho - 1) / (q-1)
    double calc_magnetization() {
        vector<int> counts(q_states, 0);
        for (int s : spins) counts[s]++;
        
        int max_count = *max_element(counts.begin(), counts.end());
        double max_rho = (double)max_count / N;
        
        return (q_states * max_rho - 1.0) / (q_states - 1.0);
    }

    int get_N() const { return N; }
    int get_q() const { return q_states; }
};

int main() {
    Config conf;
    int total_sites = conf.L * conf.L;

    string res_filename = "potts_square_q" + to_string(conf.q) + "_L" + to_string(conf.L) + ".csv";
    string log_filename = "sim_log.txt";
    
    ofstream ofs_res(res_filename);
    ofs_res << "beta,abs_m_naive,m2_improved,m4_naive,binder_cumulant" << endl;

    auto total_start = chrono::high_resolution_clock::now();
    
    PottsSquare model(conf.L, conf.q, conf.seed);
    cout << "Starting Potts (q=" << conf.q << ") on Square Lattice: L=" << conf.L << "..." << endl;

    for (int i = 0; i < conf.beta_num + 1; ++i) {
        double beta = conf.beta_min + (conf.beta_max - conf.beta_min) * (double)i / conf.beta_num;
        
        // Warmup
        for (int s = 0; s < conf.n_warmup; ++s) model.wolff_step(beta);

        // Measurement
        double sum_m = 0;
        double sum_s = 0;   // <S> 用
        double sum_m4 = 0;
        
        for (int s = 0; s < conf.n_measure; ++s) {
            int s_size = model.wolff_step(beta);
            sum_s += (double)s_size;
            
            double m = model.calc_magnetization();
            sum_m += m;
            sum_m4 += pow(m, 4);
        }

        double abs_m = sum_m / conf.n_measure;
        // PottsのM2改良推定量: <m^2> = (q-1)/q * <S>/N
        double m2_imp = ((double)(conf.q - 1) / conf.q) * (sum_s / ((double)conf.n_measure * total_sites));
        double m4_naive = sum_m4 / conf.n_measure;
        double binder = m4_naive / (m2_imp * m2_imp);

        ofs_res << fixed << setprecision(8)
                << beta << ","
                << abs_m << ","
                << m2_imp << ","
                << m4_naive << ","
                << binder << endl;
        
        cout << "Finished beta=" << fixed << setprecision(4) << beta
             << " | <m>=" << setprecision(4) << abs_m << endl;
    }
    
    auto total_end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_elapsed = total_end - total_start;
    
    // ログファイルへの一括追記
    ofstream ofs_log(log_filename, ios::app);
    ofs_log << "L=" << conf.L
            << ", beta_min=" << conf.beta_min
            << ", beta_max=" << conf.beta_max
            << ", beta_num=" << conf.beta_num
            << ", n_measure=" << conf.n_measure
            << ", total_time=" << fixed << setprecision(2) << total_elapsed.count() << "s"
            << endl;

    cout << "\nResults saved: " << res_filename << endl;
    cout << "Total time: " << total_elapsed.count() << "s" << endl;

    return 0;
}

#ifndef LSTM_H
#define LSTM_H

#include <vector>

// ========== 활성화 함수 ==========
float sigmoid(float x);
float dsigmoid(float s);
float tanhf_custom(float x);
float dtanhf(float t);

// ========== 소프트맥스 & 크로스 엔트로피 ==========
std::vector<float> softmax(const std::vector<float>& logits);
float cross_entropy(const std::vector<float>& probs, int label);

// ========== LSTM 파라미터 구조체 ==========
struct LSTMParam {
    int input_dim;
    int hidden_dim;
    int batch_size;

    std::vector<float> Wf, Wi, Wc, Wo;
    std::vector<float> bf, bi, bc, bo;
    std::vector<float> dWf, dWi, dWc, dWo;
    std::vector<float> dbf, dbi, dbc, dbo;

    LSTMParam(int in_dim, int hid_dim, int batch);
    void zero_grad();
    void update(float lr);
};

// ========== 시계열의 각 타임스텝에서의 중간 값 보관 ==========
struct LSTMCache {
    std::vector<float> f, i, o, c_tilde;
    std::vector<float> concat;
    std::vector<float> c, h;
};

// ========== 순전파 결과를 보관 ==========
struct LSTMForwardResult {
    std::vector<LSTMCache> caches;
};

// ========== 유틸리티 함수 ==========
std::vector<float> matVecMulAddBias(const std::vector<float>& W, const std::vector<float>& b, const std::vector<float>& x, int out_dim);

// ========== LSTM 순전파 & 역전파 ==========
LSTMForwardResult LSTMForwardFunc(const std::vector<std::vector<float>>& inputs, LSTMParam& param);
void LSTMBackwardFunc(const std::vector<std::vector<float>>& inputs, const LSTMForwardResult& forwardRes, std::vector<float> dLoss_dh_last, std::vector<float> dLoss_dc_last, LSTMParam& param);

#endif // LSTM_H

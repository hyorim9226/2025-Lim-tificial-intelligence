#include "lstm.h"
#include <cmath>
#include <random>
#include <algorithm>

// ========== 활성화 함수 ==========
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float dsigmoid(float s) {
    return s * (1.0f - s);
}

float tanhf_custom(float x) {
    return std::tanh(x);
}

float dtanhf(float t) {
    return 1.0f - t * t;
}

// ========== 소프트맥스 & 크로스 엔트로피 ==========
std::vector<float> softmax(const std::vector<float>& logits) {
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    float sumExp = 0.0f;
    for (auto& l : logits) {
        sumExp += std::exp(l - maxLogit);
    }
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp(logits[i] - maxLogit) / sumExp;
    }
    return probs;
}

float cross_entropy(const std::vector<float>& probs, int label) {
    float p = std::max(probs[label], 1e-7f);
    return -std::log(p);
}

// ========== LSTMParam ==========
LSTMParam::LSTMParam(int in_dim, int hid_dim, int batch)
    : input_dim(in_dim), hidden_dim(hid_dim), batch_size(batch) {
    size_t gateSize = (input_dim + hidden_dim) * hidden_dim;
    Wf.resize(gateSize);
    Wi.resize(gateSize);
    Wc.resize(gateSize);
    Wo.resize(gateSize);

    bf.resize(hidden_dim);
    bi.resize(hidden_dim);
    bc.resize(hidden_dim);
    bo.resize(hidden_dim);

    dWf.resize(gateSize, 0.0f);
    dWi.resize(gateSize, 0.0f);
    dWc.resize(gateSize, 0.0f);
    dWo.resize(gateSize, 0.0f);

    dbf.resize(hidden_dim, 0.0f);
    dbi.resize(hidden_dim, 0.0f);
    dbc.resize(hidden_dim, 0.0f);
    dbo.resize(hidden_dim, 0.0f);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    auto initFunc = [&](std::vector<float>& vec) {
        for (auto& v : vec) {
            v = dist(gen);
        }
    };

    initFunc(Wf);
    initFunc(Wi);
    initFunc(Wc);
    initFunc(Wo);
    initFunc(bf);
    initFunc(bi);
    initFunc(bc);
    initFunc(bo);
}

void LSTMParam::zero_grad() {
    std::fill(dWf.begin(), dWf.end(), 0.0f);
    std::fill(dWi.begin(), dWi.end(), 0.0f);
    std::fill(dWc.begin(), dWc.end(), 0.0f);
    std::fill(dWo.begin(), dWo.end(), 0.0f);
    std::fill(dbf.begin(), dbf.end(), 0.0f);
    std::fill(dbi.begin(), dbi.end(), 0.0f);
    std::fill(dbc.begin(), dbc.end(), 0.0f);
    std::fill(dbo.begin(), dbo.end(), 0.0f);
}

void LSTMParam::update(float lr) {
    for (size_t i = 0; i < Wf.size(); i++) {
        Wf[i] -= lr * dWf[i];
        Wi[i] -= lr * dWi[i];
        Wc[i] -= lr * dWc[i];
        Wo[i] -= lr * dWo[i];
    }
    for (size_t i = 0; i < bf.size(); i++) {
        bf[i] -= lr * dbf[i];
        bi[i] -= lr * dbi[i];
        bc[i] -= lr * dbc[i];
        bo[i] -= lr * dbo[i];
    }
}

// ========== 유틸리티 함수 ==========
std::vector<float> matVecMulAddBias(const std::vector<float>& W, const std::vector<float>& b, const std::vector<float>& x, int out_dim) {
    std::vector<float> result(out_dim, 0.0f);
    int in_dim = (int)x.size();
    for (int j = 0; j < out_dim; j++) {
        float sumVal = 0.f;
        for (int i = 0; i < in_dim; i++) {
            sumVal += W[i * out_dim + j] * x[i];
        }
        result[j] = sumVal + b[j];
    }
    return result;
}

// ========== LSTM 순전파 ==========
LSTMForwardResult LSTMForwardFunc(const std::vector<std::vector<float>>& inputs, LSTMParam& param) {
    int seqLen = (int)inputs.size();
    int input_dim = param.input_dim;
    int hidden_dim = param.hidden_dim;

    LSTMForwardResult result;
    result.caches.resize(seqLen);

    std::vector<float> h_prev(hidden_dim, 0.0f);
    std::vector<float> c_prev(hidden_dim, 0.0f);

    for (int t = 0; t < seqLen; t++) {
        std::vector<float> concat(hidden_dim + input_dim);
        for (int i = 0; i < hidden_dim; i++) {
            concat[i] = h_prev[i];
        }
        for (int i = 0; i < input_dim; i++) {
            concat[hidden_dim + i] = inputs[t][i];
        }

        std::vector<float> f_t = matVecMulAddBias(param.Wf, param.bf, concat, hidden_dim);
        std::vector<float> i_t = matVecMulAddBias(param.Wi, param.bi, concat, hidden_dim);
        std::vector<float> c_tilde_t = matVecMulAddBias(param.Wc, param.bc, concat, hidden_dim);
        std::vector<float> o_t = matVecMulAddBias(param.Wo, param.bo, concat, hidden_dim);

        for (int j = 0; j < hidden_dim; j++) {
            f_t[j] = sigmoid(f_t[j]);
            i_t[j] = sigmoid(i_t[j]);
            c_tilde_t[j] = tanhf_custom(c_tilde_t[j]);
            o_t[j] = sigmoid(o_t[j]);
        }

        std::vector<float> c_t(hidden_dim, 0.0f);
        std::vector<float> h_t(hidden_dim, 0.0f);
        for (int j = 0; j < hidden_dim; j++) {
            c_t[j] = f_t[j] * c_prev[j] + i_t[j] * c_tilde_t[j];
            h_t[j] = o_t[j] * tanhf_custom(c_t[j]);
        }

        result.caches[t].f = f_t;
        result.caches[t].i = i_t;
        result.caches[t].o = o_t;
        result.caches[t].c_tilde = c_tilde_t;
        result.caches[t].concat = concat;
        result.caches[t].c = c_t;
        result.caches[t].h = h_t;

        h_prev = h_t;
        c_prev = c_t;
    }

    return result;
}


// ========== LSTM 역전파 ==========
void LSTMBackwardFunc(const std::vector<std::vector<float>>& inputs, const LSTMForwardResult& forwardRes, std::vector<float> dLoss_dh_last, std::vector<float> dLoss_dc_last, LSTMParam& param) {
        int seqLen = (int)inputs.size();
    int hidden_dim = param.hidden_dim;
    int input_dim = param.input_dim;

    std::vector<float> dh_next = dLoss_dh_last;
    std::vector<float> dc_next = dLoss_dc_last;

    for (int t = seqLen - 1; t >= 0; t--) {
        const auto& cache = forwardRes.caches[t];
        const auto& f_t = cache.f;
        const auto& i_t = cache.i;
        const auto& o_t = cache.o;
        const auto& c_tilde_t = cache.c_tilde;
        const auto& c_t = cache.c;
        const auto& h_t = cache.h;
        const auto& concat = cache.concat;

        std::vector<float> dh_t(hidden_dim, 0.0f);
        for (int j = 0; j < hidden_dim; j++) {
            dh_t[j] = dh_next[j];
        }

        std::vector<float> dc_t(hidden_dim, 0.0f);
        for (int j = 0; j < hidden_dim; j++) {
            float d_tanh_c_t = dtanhf(std::tanh(c_t[j]));
            dc_t[j] = dc_next[j] + dh_t[j] * o_t[j] * d_tanh_c_t;
        }

        std::vector<float> df_t(hidden_dim), di_t(hidden_dim),
            dc_tilde_t(hidden_dim), do_t(hidden_dim);

        const std::vector<float>& c_prev = (t == 0) ?
            std::vector<float>(hidden_dim, 0.0f) : forwardRes.caches[t - 1].c;

        for (int j = 0; j < hidden_dim; j++) {
            df_t[j] = dc_t[j] * c_prev[j] * dsigmoid(f_t[j]);
            di_t[j] = dc_t[j] * c_tilde_t[j] * dsigmoid(i_t[j]);
            dc_tilde_t[j] = dc_t[j] * i_t[j] * dtanhf(c_tilde_t[j]);
            do_t[j] = dh_t[j] * std::tanh(c_t[j]) * dsigmoid(o_t[j]);
        }

        for (int j = 0; j < hidden_dim; j++) {
            dc_next[j] = dc_t[j] * f_t[j];
        }

        std::vector<float> dconcat(hidden_dim + input_dim, 0.0f);

        auto addMatVecMulT = [&](const std::vector<float>& W, const std::vector<float>& dGate) {
            for (int i = 0; i < hidden_dim + input_dim; i++) {
                float sumVal = 0.f;
                for (int j = 0; j < hidden_dim; j++) {
                    sumVal += W[i * hidden_dim + j] * dGate[j];
                }
                dconcat[i] += sumVal;
            }
        };

        addMatVecMulT(param.Wf, df_t);
        addMatVecMulT(param.Wi, di_t);
        addMatVecMulT(param.Wc, dc_tilde_t);
        addMatVecMulT(param.Wo, do_t);

        for (int j = 0; j < hidden_dim; j++) {
            dh_next[j] = dconcat[j];
        }

        auto accumulateMatGrad = [&](std::vector<float>& dW,
                                     std::vector<float>& db,
                                     const std::vector<float>& gate_grad) {
            for (int i = 0; i < hidden_dim + input_dim; i++) {
                for (int j = 0; j < hidden_dim; j++) {
                    dW[i * hidden_dim + j] += concat[i] * gate_grad[j];
                    }
                }
                for (int j = 0; j < hidden_dim; j++) {
                    db[j] += gate_grad[j];
                }
            };

        accumulateMatGrad(param.dWf, param.bf, df_t);
        accumulateMatGrad(param.dWi, param.bi, di_t);
        accumulateMatGrad(param.dWc, param.bc, dc_tilde_t);
        accumulateMatGrad(param.dWo, param.bo, do_t);
    }
}

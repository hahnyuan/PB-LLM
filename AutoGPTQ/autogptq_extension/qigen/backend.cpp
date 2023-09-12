 #include <torch/all.h>
 #include <torch/python.h>
 #include <omp.h>
 #include <cmath>
 #include <immintrin.h>
 
 #define mymin(a,b) ((a)<(b)?(a):(b))
 #define mymax(a,b) ((a)>(b)?(a):(b))
  void quantize_scalar(float* A, int* BQ, float* scales, float* zeros, int n, int m, int bits){ 
 	//find scales and zeros arrays 
 	//quantize 
 	int pack = 32/bits;
 	for (int j = 0; j < m; j++){
 		for (int i = 0; i < n; i+=pack){
 			uint32_t acc = 0;
 			for (int ii = i; ii < i+pack; ii++){
 				float ftemp = std::round((A[ii*m+j] + zeros[j])/scales[j]);
 				int temp = (int)ftemp;
 				acc = acc | (temp << (bits*(ii-i)));
 			}
 			BQ[(i/pack)*m+j] = acc;
 			//BQ[0] = acc;
 		}
 	}
 }
 
 void quant_scalar_cpu(
 	torch::Tensor in, torch::Tensor out, 
 	torch::Tensor scales, torch::Tensor zeros, int bits
 ) {
 
 	int N  = in.size(0);
 	int M  = in.size(1);
 
 	float* input = in.data_ptr<float>(); 
 	float* s   = scales.data_ptr<float>();
 	float* z   = zeros.data_ptr<float>();
 	int* O   = out.data_ptr<int>();
 		
 	quantize_scalar(input, O, s, z, N, M, bits);
 
 }
void compute_reduction_cpu(const float* in, float* out, int n, int m, int gs){
#pragma omp parallel num_threads(12)
{
#pragma omp for collapse(2)
for(int i = 0; i < n; i++){
for(int j0 = 0; j0 < m; j0+=gs){
__m256 acc = _mm256_setzero_ps();
for(int j1 = j0; j1 < j0+gs; j1+=8){
__m256 x = _mm256_loadu_ps(&in[i*m  + j1]);
acc = _mm256_add_ps(acc, x);
}
const __m128 hiQuad = _mm256_extractf128_ps(acc, 1);
const __m128 loQuad = _mm256_castps256_ps128(acc);
const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
const __m128 sumDual = _mm_add_ps(sumQuad, hiDual);
const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
const __m128 sum = _mm_add_ss(hi, sumDual);
out[(i*m + j0)/gs] = _mm_cvtss_f32(sum);
}
}
}
}
void compute_reduction(torch::Tensor in, torch::Tensor out, int N, int M, int gs){
float* I = in.data_ptr<float>();
float* O = out.data_ptr<float>();
compute_reduction_cpu(I, O, N, M, gs);
}
void unquantize_sim_cpu(const int* in, float* out, float* s, float* z, int n, int m, int bits, int gs){
#pragma omp parallel num_threads(12)
{
int packed = 32/bits;
int mask = (1<<bits) - 1;
#pragma omp for
for(int i0 = 0; i0 < n; i0+=gs){
int row = i0 / gs;
for(int i1 = i0; i1 < i0+gs; i1+=packed){
for(int j0 = 0; j0 < m; j0++){
for(int k = 0; k < packed; k++){
out[(i1+k)*m + j0] = ((float)((in[i1*m/packed + j0] >> (bits*k)) & mask) - z[(row)*m + j0]) * s[(row)*m + j0];
}
}
}
}
}
}
void unquantize_sim(torch::Tensor in, torch::Tensor out, torch::Tensor s, torch::Tensor z, int N, int M, int bits, int gs){
int* I = in.data_ptr<int>();
float* O = out.data_ptr<float>();
float* S = s.data_ptr<float>();
float* Z = z.data_ptr<float>();
unquantize_sim_cpu(I, O, S, Z, N, M, bits, gs);
}
inline
void q2gemm(const float* __restrict__ input, 
const int* __restrict__ W, 
const float* __restrict__ scales, 
const float* __restrict__ zeros, 
const float* __restrict__ bias, 
 const float* __restrict__ sums, 
 float* __restrict__ output,
const int n,
const int m,
const int t,
const int nb,
const int mb,
const int tb,
int ogtt,
const int cutoff){
#pragma omp parallel num_threads(12)
{
int tid;
const int mu = 16;
const int nu = 1;
const int tu = 32;
const int on = n / nb;
const int om = m / mb;
const __m256i mask = _mm256_set1_epi32(3);
tid = omp_get_thread_num();
int tt = ogtt;
if(tid >= cutoff){
tt -= tb;
}
const int base_output = tid >= cutoff ?
 (tid-cutoff)*tt + (tt+tb)*cutoff: 
 tid*tt;
const int base_W = tid >= cutoff ?
 ((tid-cutoff)*tt + (tt+tb)*cutoff)*m/16: 
 tid*tt*m/16;
for(int j = 0; j < tt; j+=tb){
for(int i = 0; i < on; i++) {
for(int k = 0; k < om; k++) {
for(int i1 = 0; i1 < nb; i1+=nu) {
int j1 = 0;
for(; j1 < tb-tu+1; j1+=tu) {
__m256 acc0_0 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+0]);
__m256 acc0_8 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+8]);
__m256 acc0_16 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+16]);
__m256 acc0_24 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+24]);
for(int k1 = 0; k1 < mb; k1+=mu) {
for(int k2 = k1; k2 < k1+mu; k2+=16){
__m256i w0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/16 + k*mb*tb/16 + k2*tb/16 + j1+0]);
__m256i w8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/16 + k*mb*tb/16 + k2*tb/16 + j1+8]);
__m256i w16 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/16 + k*mb*tb/16 + k2*tb/16 + j1+16]);
__m256i w24 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/16 + k*mb*tb/16 + k2*tb/16 + j1+24]);
__m256 v0_15 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+15)*nb + i1+0]);
__m256i ws0_15 = _mm256_srli_epi32(w0, 30);
__m256i ws8_15 = _mm256_srli_epi32(w8, 30);
__m256i ws16_15 = _mm256_srli_epi32(w16, 30);
__m256i ws24_15 = _mm256_srli_epi32(w24, 30);
__m256i wsa0_15= _mm256_and_si256(ws0_15, mask);
__m256i wsa8_15= _mm256_and_si256(ws8_15, mask);
__m256i wsa16_15= _mm256_and_si256(ws16_15, mask);
__m256i wsa24_15= _mm256_and_si256(ws24_15, mask);
__m256 l0_15 = _mm256_cvtepi32_ps(wsa0_15);
__m256 l8_15 = _mm256_cvtepi32_ps(wsa8_15);
__m256 l16_15 = _mm256_cvtepi32_ps(wsa16_15);
__m256 l24_15 = _mm256_cvtepi32_ps(wsa24_15);
acc0_0 = _mm256_fmadd_ps(v0_15, l0_15, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_15, l8_15, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_15, l16_15, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_15, l24_15, acc0_24);
__m256 v0_14 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+14)*nb + i1+0]);
__m256i ws0_14 = _mm256_srli_epi32(w0, 28);
__m256i ws8_14 = _mm256_srli_epi32(w8, 28);
__m256i ws16_14 = _mm256_srli_epi32(w16, 28);
__m256i ws24_14 = _mm256_srli_epi32(w24, 28);
__m256i wsa0_14= _mm256_and_si256(ws0_14, mask);
__m256i wsa8_14= _mm256_and_si256(ws8_14, mask);
__m256i wsa16_14= _mm256_and_si256(ws16_14, mask);
__m256i wsa24_14= _mm256_and_si256(ws24_14, mask);
__m256 l0_14 = _mm256_cvtepi32_ps(wsa0_14);
__m256 l8_14 = _mm256_cvtepi32_ps(wsa8_14);
__m256 l16_14 = _mm256_cvtepi32_ps(wsa16_14);
__m256 l24_14 = _mm256_cvtepi32_ps(wsa24_14);
acc0_0 = _mm256_fmadd_ps(v0_14, l0_14, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_14, l8_14, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_14, l16_14, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_14, l24_14, acc0_24);
__m256 v0_13 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+13)*nb + i1+0]);
__m256i ws0_13 = _mm256_srli_epi32(w0, 26);
__m256i ws8_13 = _mm256_srli_epi32(w8, 26);
__m256i ws16_13 = _mm256_srli_epi32(w16, 26);
__m256i ws24_13 = _mm256_srli_epi32(w24, 26);
__m256i wsa0_13= _mm256_and_si256(ws0_13, mask);
__m256i wsa8_13= _mm256_and_si256(ws8_13, mask);
__m256i wsa16_13= _mm256_and_si256(ws16_13, mask);
__m256i wsa24_13= _mm256_and_si256(ws24_13, mask);
__m256 l0_13 = _mm256_cvtepi32_ps(wsa0_13);
__m256 l8_13 = _mm256_cvtepi32_ps(wsa8_13);
__m256 l16_13 = _mm256_cvtepi32_ps(wsa16_13);
__m256 l24_13 = _mm256_cvtepi32_ps(wsa24_13);
acc0_0 = _mm256_fmadd_ps(v0_13, l0_13, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_13, l8_13, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_13, l16_13, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_13, l24_13, acc0_24);
__m256 v0_12 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+12)*nb + i1+0]);
__m256i ws0_12 = _mm256_srli_epi32(w0, 24);
__m256i ws8_12 = _mm256_srli_epi32(w8, 24);
__m256i ws16_12 = _mm256_srli_epi32(w16, 24);
__m256i ws24_12 = _mm256_srli_epi32(w24, 24);
__m256i wsa0_12= _mm256_and_si256(ws0_12, mask);
__m256i wsa8_12= _mm256_and_si256(ws8_12, mask);
__m256i wsa16_12= _mm256_and_si256(ws16_12, mask);
__m256i wsa24_12= _mm256_and_si256(ws24_12, mask);
__m256 l0_12 = _mm256_cvtepi32_ps(wsa0_12);
__m256 l8_12 = _mm256_cvtepi32_ps(wsa8_12);
__m256 l16_12 = _mm256_cvtepi32_ps(wsa16_12);
__m256 l24_12 = _mm256_cvtepi32_ps(wsa24_12);
acc0_0 = _mm256_fmadd_ps(v0_12, l0_12, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_12, l8_12, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_12, l16_12, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_12, l24_12, acc0_24);
__m256 v0_11 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+11)*nb + i1+0]);
__m256i ws0_11 = _mm256_srli_epi32(w0, 22);
__m256i ws8_11 = _mm256_srli_epi32(w8, 22);
__m256i ws16_11 = _mm256_srli_epi32(w16, 22);
__m256i ws24_11 = _mm256_srli_epi32(w24, 22);
__m256i wsa0_11= _mm256_and_si256(ws0_11, mask);
__m256i wsa8_11= _mm256_and_si256(ws8_11, mask);
__m256i wsa16_11= _mm256_and_si256(ws16_11, mask);
__m256i wsa24_11= _mm256_and_si256(ws24_11, mask);
__m256 l0_11 = _mm256_cvtepi32_ps(wsa0_11);
__m256 l8_11 = _mm256_cvtepi32_ps(wsa8_11);
__m256 l16_11 = _mm256_cvtepi32_ps(wsa16_11);
__m256 l24_11 = _mm256_cvtepi32_ps(wsa24_11);
acc0_0 = _mm256_fmadd_ps(v0_11, l0_11, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_11, l8_11, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_11, l16_11, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_11, l24_11, acc0_24);
__m256 v0_10 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+10)*nb + i1+0]);
__m256i ws0_10 = _mm256_srli_epi32(w0, 20);
__m256i ws8_10 = _mm256_srli_epi32(w8, 20);
__m256i ws16_10 = _mm256_srli_epi32(w16, 20);
__m256i ws24_10 = _mm256_srli_epi32(w24, 20);
__m256i wsa0_10= _mm256_and_si256(ws0_10, mask);
__m256i wsa8_10= _mm256_and_si256(ws8_10, mask);
__m256i wsa16_10= _mm256_and_si256(ws16_10, mask);
__m256i wsa24_10= _mm256_and_si256(ws24_10, mask);
__m256 l0_10 = _mm256_cvtepi32_ps(wsa0_10);
__m256 l8_10 = _mm256_cvtepi32_ps(wsa8_10);
__m256 l16_10 = _mm256_cvtepi32_ps(wsa16_10);
__m256 l24_10 = _mm256_cvtepi32_ps(wsa24_10);
acc0_0 = _mm256_fmadd_ps(v0_10, l0_10, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_10, l8_10, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_10, l16_10, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_10, l24_10, acc0_24);
__m256 v0_9 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+9)*nb + i1+0]);
__m256i ws0_9 = _mm256_srli_epi32(w0, 18);
__m256i ws8_9 = _mm256_srli_epi32(w8, 18);
__m256i ws16_9 = _mm256_srli_epi32(w16, 18);
__m256i ws24_9 = _mm256_srli_epi32(w24, 18);
__m256i wsa0_9= _mm256_and_si256(ws0_9, mask);
__m256i wsa8_9= _mm256_and_si256(ws8_9, mask);
__m256i wsa16_9= _mm256_and_si256(ws16_9, mask);
__m256i wsa24_9= _mm256_and_si256(ws24_9, mask);
__m256 l0_9 = _mm256_cvtepi32_ps(wsa0_9);
__m256 l8_9 = _mm256_cvtepi32_ps(wsa8_9);
__m256 l16_9 = _mm256_cvtepi32_ps(wsa16_9);
__m256 l24_9 = _mm256_cvtepi32_ps(wsa24_9);
acc0_0 = _mm256_fmadd_ps(v0_9, l0_9, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_9, l8_9, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_9, l16_9, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_9, l24_9, acc0_24);
__m256 v0_8 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+8)*nb + i1+0]);
__m256i ws0_8 = _mm256_srli_epi32(w0, 16);
__m256i ws8_8 = _mm256_srli_epi32(w8, 16);
__m256i ws16_8 = _mm256_srli_epi32(w16, 16);
__m256i ws24_8 = _mm256_srli_epi32(w24, 16);
__m256i wsa0_8= _mm256_and_si256(ws0_8, mask);
__m256i wsa8_8= _mm256_and_si256(ws8_8, mask);
__m256i wsa16_8= _mm256_and_si256(ws16_8, mask);
__m256i wsa24_8= _mm256_and_si256(ws24_8, mask);
__m256 l0_8 = _mm256_cvtepi32_ps(wsa0_8);
__m256 l8_8 = _mm256_cvtepi32_ps(wsa8_8);
__m256 l16_8 = _mm256_cvtepi32_ps(wsa16_8);
__m256 l24_8 = _mm256_cvtepi32_ps(wsa24_8);
acc0_0 = _mm256_fmadd_ps(v0_8, l0_8, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_8, l8_8, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_8, l16_8, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_8, l24_8, acc0_24);
__m256 v0_7 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+7)*nb + i1+0]);
__m256i ws0_7 = _mm256_srli_epi32(w0, 14);
__m256i ws8_7 = _mm256_srli_epi32(w8, 14);
__m256i ws16_7 = _mm256_srli_epi32(w16, 14);
__m256i ws24_7 = _mm256_srli_epi32(w24, 14);
__m256i wsa0_7= _mm256_and_si256(ws0_7, mask);
__m256i wsa8_7= _mm256_and_si256(ws8_7, mask);
__m256i wsa16_7= _mm256_and_si256(ws16_7, mask);
__m256i wsa24_7= _mm256_and_si256(ws24_7, mask);
__m256 l0_7 = _mm256_cvtepi32_ps(wsa0_7);
__m256 l8_7 = _mm256_cvtepi32_ps(wsa8_7);
__m256 l16_7 = _mm256_cvtepi32_ps(wsa16_7);
__m256 l24_7 = _mm256_cvtepi32_ps(wsa24_7);
acc0_0 = _mm256_fmadd_ps(v0_7, l0_7, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_7, l8_7, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_7, l16_7, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_7, l24_7, acc0_24);
__m256 v0_6 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+6)*nb + i1+0]);
__m256i ws0_6 = _mm256_srli_epi32(w0, 12);
__m256i ws8_6 = _mm256_srli_epi32(w8, 12);
__m256i ws16_6 = _mm256_srli_epi32(w16, 12);
__m256i ws24_6 = _mm256_srli_epi32(w24, 12);
__m256i wsa0_6= _mm256_and_si256(ws0_6, mask);
__m256i wsa8_6= _mm256_and_si256(ws8_6, mask);
__m256i wsa16_6= _mm256_and_si256(ws16_6, mask);
__m256i wsa24_6= _mm256_and_si256(ws24_6, mask);
__m256 l0_6 = _mm256_cvtepi32_ps(wsa0_6);
__m256 l8_6 = _mm256_cvtepi32_ps(wsa8_6);
__m256 l16_6 = _mm256_cvtepi32_ps(wsa16_6);
__m256 l24_6 = _mm256_cvtepi32_ps(wsa24_6);
acc0_0 = _mm256_fmadd_ps(v0_6, l0_6, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_6, l8_6, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_6, l16_6, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_6, l24_6, acc0_24);
__m256 v0_5 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+5)*nb + i1+0]);
__m256i ws0_5 = _mm256_srli_epi32(w0, 10);
__m256i ws8_5 = _mm256_srli_epi32(w8, 10);
__m256i ws16_5 = _mm256_srli_epi32(w16, 10);
__m256i ws24_5 = _mm256_srli_epi32(w24, 10);
__m256i wsa0_5= _mm256_and_si256(ws0_5, mask);
__m256i wsa8_5= _mm256_and_si256(ws8_5, mask);
__m256i wsa16_5= _mm256_and_si256(ws16_5, mask);
__m256i wsa24_5= _mm256_and_si256(ws24_5, mask);
__m256 l0_5 = _mm256_cvtepi32_ps(wsa0_5);
__m256 l8_5 = _mm256_cvtepi32_ps(wsa8_5);
__m256 l16_5 = _mm256_cvtepi32_ps(wsa16_5);
__m256 l24_5 = _mm256_cvtepi32_ps(wsa24_5);
acc0_0 = _mm256_fmadd_ps(v0_5, l0_5, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_5, l8_5, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_5, l16_5, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_5, l24_5, acc0_24);
__m256 v0_4 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+4)*nb + i1+0]);
__m256i ws0_4 = _mm256_srli_epi32(w0, 8);
__m256i ws8_4 = _mm256_srli_epi32(w8, 8);
__m256i ws16_4 = _mm256_srli_epi32(w16, 8);
__m256i ws24_4 = _mm256_srli_epi32(w24, 8);
__m256i wsa0_4= _mm256_and_si256(ws0_4, mask);
__m256i wsa8_4= _mm256_and_si256(ws8_4, mask);
__m256i wsa16_4= _mm256_and_si256(ws16_4, mask);
__m256i wsa24_4= _mm256_and_si256(ws24_4, mask);
__m256 l0_4 = _mm256_cvtepi32_ps(wsa0_4);
__m256 l8_4 = _mm256_cvtepi32_ps(wsa8_4);
__m256 l16_4 = _mm256_cvtepi32_ps(wsa16_4);
__m256 l24_4 = _mm256_cvtepi32_ps(wsa24_4);
acc0_0 = _mm256_fmadd_ps(v0_4, l0_4, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_4, l8_4, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_4, l16_4, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_4, l24_4, acc0_24);
__m256 v0_3 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+3)*nb + i1+0]);
__m256i ws0_3 = _mm256_srli_epi32(w0, 6);
__m256i ws8_3 = _mm256_srli_epi32(w8, 6);
__m256i ws16_3 = _mm256_srli_epi32(w16, 6);
__m256i ws24_3 = _mm256_srli_epi32(w24, 6);
__m256i wsa0_3= _mm256_and_si256(ws0_3, mask);
__m256i wsa8_3= _mm256_and_si256(ws8_3, mask);
__m256i wsa16_3= _mm256_and_si256(ws16_3, mask);
__m256i wsa24_3= _mm256_and_si256(ws24_3, mask);
__m256 l0_3 = _mm256_cvtepi32_ps(wsa0_3);
__m256 l8_3 = _mm256_cvtepi32_ps(wsa8_3);
__m256 l16_3 = _mm256_cvtepi32_ps(wsa16_3);
__m256 l24_3 = _mm256_cvtepi32_ps(wsa24_3);
acc0_0 = _mm256_fmadd_ps(v0_3, l0_3, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_3, l8_3, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_3, l16_3, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_3, l24_3, acc0_24);
__m256 v0_2 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+2)*nb + i1+0]);
__m256i ws0_2 = _mm256_srli_epi32(w0, 4);
__m256i ws8_2 = _mm256_srli_epi32(w8, 4);
__m256i ws16_2 = _mm256_srli_epi32(w16, 4);
__m256i ws24_2 = _mm256_srli_epi32(w24, 4);
__m256i wsa0_2= _mm256_and_si256(ws0_2, mask);
__m256i wsa8_2= _mm256_and_si256(ws8_2, mask);
__m256i wsa16_2= _mm256_and_si256(ws16_2, mask);
__m256i wsa24_2= _mm256_and_si256(ws24_2, mask);
__m256 l0_2 = _mm256_cvtepi32_ps(wsa0_2);
__m256 l8_2 = _mm256_cvtepi32_ps(wsa8_2);
__m256 l16_2 = _mm256_cvtepi32_ps(wsa16_2);
__m256 l24_2 = _mm256_cvtepi32_ps(wsa24_2);
acc0_0 = _mm256_fmadd_ps(v0_2, l0_2, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_2, l8_2, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_2, l16_2, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_2, l24_2, acc0_24);
__m256 v0_1 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+1)*nb + i1+0]);
__m256i ws0_1 = _mm256_srli_epi32(w0, 2);
__m256i ws8_1 = _mm256_srli_epi32(w8, 2);
__m256i ws16_1 = _mm256_srli_epi32(w16, 2);
__m256i ws24_1 = _mm256_srli_epi32(w24, 2);
__m256i wsa0_1= _mm256_and_si256(ws0_1, mask);
__m256i wsa8_1= _mm256_and_si256(ws8_1, mask);
__m256i wsa16_1= _mm256_and_si256(ws16_1, mask);
__m256i wsa24_1= _mm256_and_si256(ws24_1, mask);
__m256 l0_1 = _mm256_cvtepi32_ps(wsa0_1);
__m256 l8_1 = _mm256_cvtepi32_ps(wsa8_1);
__m256 l16_1 = _mm256_cvtepi32_ps(wsa16_1);
__m256 l24_1 = _mm256_cvtepi32_ps(wsa24_1);
acc0_0 = _mm256_fmadd_ps(v0_1, l0_1, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_1, l8_1, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_1, l16_1, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_1, l24_1, acc0_24);
__m256 v0_0 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+0)*nb + i1+0]);
__m256i ws0_0 = _mm256_srli_epi32(w0, 0);
__m256i ws8_0 = _mm256_srli_epi32(w8, 0);
__m256i ws16_0 = _mm256_srli_epi32(w16, 0);
__m256i ws24_0 = _mm256_srli_epi32(w24, 0);
__m256i wsa0_0= _mm256_and_si256(ws0_0, mask);
__m256i wsa8_0= _mm256_and_si256(ws8_0, mask);
__m256i wsa16_0= _mm256_and_si256(ws16_0, mask);
__m256i wsa24_0= _mm256_and_si256(ws24_0, mask);
__m256 l0_0 = _mm256_cvtepi32_ps(wsa0_0);
__m256 l8_0 = _mm256_cvtepi32_ps(wsa8_0);
__m256 l16_0 = _mm256_cvtepi32_ps(wsa16_0);
__m256 l24_0 = _mm256_cvtepi32_ps(wsa24_0);
acc0_0 = _mm256_fmadd_ps(v0_0, l0_0, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_0, l8_0, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_0, l16_0, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_0, l24_0, acc0_24);
}
}
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+0], acc0_0);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+8], acc0_8);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+16], acc0_16);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+24], acc0_24);
}
}
}
}
}
#pragma omp barrier
for (int i = 0; i < n; i++) {
__m256 r = _mm256_set1_ps(sums[i]);
for (int j = 0; j < tt; j+=32){
__m256 o0 = _mm256_loadu_ps(&output[i*t + base_output + j + 0]);
__m256 o8 = _mm256_loadu_ps(&output[i*t + base_output + j + 8]);
__m256 o16 = _mm256_loadu_ps(&output[i*t + base_output + j + 16]);
__m256 o24 = _mm256_loadu_ps(&output[i*t + base_output + j + 24]);
__m256 z0 = _mm256_loadu_ps(&zeros[base_output + j + 0]);
__m256 z8 = _mm256_loadu_ps(&zeros[base_output + j + 8]);
__m256 z16 = _mm256_loadu_ps(&zeros[base_output + j + 16]);
__m256 z24 = _mm256_loadu_ps(&zeros[base_output + j + 24]);
__m256 b0 = _mm256_loadu_ps(&bias[base_output + j + 0]);
__m256 b8 = _mm256_loadu_ps(&bias[base_output + j + 8]);
__m256 b16 = _mm256_loadu_ps(&bias[base_output + j + 16]);
__m256 b24 = _mm256_loadu_ps(&bias[base_output + j + 24]);
__m256 s0 = _mm256_loadu_ps(&scales[base_output + j + 0]);
__m256 s8 = _mm256_loadu_ps(&scales[base_output + j + 8]);
__m256 s16 = _mm256_loadu_ps(&scales[base_output + j + 16]);
__m256 s24 = _mm256_loadu_ps(&scales[base_output + j + 24]);
__m256 zr0 = _mm256_fnmadd_ps(z0, r, o0);
__m256 zr8 = _mm256_fnmadd_ps(z8, r, o8);
__m256 zr16 = _mm256_fnmadd_ps(z16, r, o16);
__m256 zr24 = _mm256_fnmadd_ps(z24, r, o24);
__m256 o20 = _mm256_fmadd_ps(zr0, s0, b0);
__m256 o28 = _mm256_fmadd_ps(zr8, s8, b8);
__m256 o216 = _mm256_fmadd_ps(zr16, s16, b16);
__m256 o224 = _mm256_fmadd_ps(zr24, s24, b24);
_mm256_storeu_ps(&output[i*t + base_output + j + 0], o20);
_mm256_storeu_ps(&output[i*t + base_output + j + 8], o28);
_mm256_storeu_ps(&output[i*t + base_output + j + 16], o216);
_mm256_storeu_ps(&output[i*t + base_output + j + 24], o224);
}
}
}
}
inline void forward2_cpu(
torch::Tensor in, torch::Tensor weight, torch::Tensor out,
torch::Tensor bias, torch::Tensor scales, torch::Tensor zeros, torch::Tensor sums,
int N, int M, int T, int nb, int mb, int tb, int tt, int cutoff){
int*   W = weight.data_ptr<int>();
float* input = in.data_ptr<float>();
float* b   = bias.data_ptr<float>();
float* s   = scales.data_ptr<float>();
float* z   = zeros.data_ptr<float>();
float* r   = sums.data_ptr<float>();
float* O   = out.data_ptr<float>();

q2gemm(input, W, s, z, b, r, O, N, M, T, nb, mb, tb, tt, cutoff);
}
inline
void q2gemm_gs(const float* __restrict__ input, 
const int* __restrict__ W, 
const float* __restrict__ scales, 
const float* __restrict__ zeros, 
const float* __restrict__ bias, 
 const float* __restrict__ sums, 
 float* __restrict__ output,
const int n,
const int m,
const int t,
const int nb,
const int mb,
const int tb,
int ogtt,
const int gs,
const int cutoff){
#pragma omp parallel num_threads(12)
{
int tid;
const int mu = 16;
const int nu = 1;
const int tu = 32;
const int on = n / nb;
const int om = m / mb;
const __m256i mask = _mm256_set1_epi32(3);
tid = omp_get_thread_num();
int tt = ogtt;
if(tid >= cutoff){
tt -= tb;
}
const int base_output = tid >= cutoff ?
 (tid-cutoff)*tt + (tt+tb)*cutoff: 
 tid*tt;
const int base_W = tid >= cutoff ?
 ((tid-cutoff)*tt + (tt+tb)*cutoff)*m/16: 
 tid*tt*m/16;
for(int j = 0; j < tt; j+=tb){
for(int i = 0; i < on; i++) {
for(int k = 0; k < om; k++) {
for(int i1 = 0; i1 < nb; i1+=nu) {
int j1 = 0;
for(; j1 < tb-tu+1; j1+=tu) {
for(int k1 = 0; k1 < mb; k1+=gs) {
__m256 acc0_0 = _mm256_setzero_ps();
__m256 acc0_8 = _mm256_setzero_ps();
__m256 acc0_16 = _mm256_setzero_ps();
__m256 acc0_24 = _mm256_setzero_ps();
for(int k2 = k1; k2 < k1+gs; k2+=16)
{
__m256i w0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/16 + k*mb*tb/16 + k2*tb/16 + j1+0]);
__m256i w8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/16 + k*mb*tb/16 + k2*tb/16 + j1+8]);
__m256i w16 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/16 + k*mb*tb/16 + k2*tb/16 + j1+16]);
__m256i w24 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/16 + k*mb*tb/16 + k2*tb/16 + j1+24]);
__m256 v0_15 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+15)*nb + i1+0]);
__m256i ws0_15 = _mm256_srli_epi32(w0, 30);
__m256i ws8_15 = _mm256_srli_epi32(w8, 30);
__m256i ws16_15 = _mm256_srli_epi32(w16, 30);
__m256i ws24_15 = _mm256_srli_epi32(w24, 30);
__m256i wsa0_15= _mm256_and_si256(ws0_15, mask);
__m256i wsa8_15= _mm256_and_si256(ws8_15, mask);
__m256i wsa16_15= _mm256_and_si256(ws16_15, mask);
__m256i wsa24_15= _mm256_and_si256(ws24_15, mask);
__m256 l0_15 = _mm256_cvtepi32_ps(wsa0_15);
__m256 l8_15 = _mm256_cvtepi32_ps(wsa8_15);
__m256 l16_15 = _mm256_cvtepi32_ps(wsa16_15);
__m256 l24_15 = _mm256_cvtepi32_ps(wsa24_15);
acc0_0 = _mm256_fmadd_ps(v0_15, l0_15, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_15, l8_15, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_15, l16_15, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_15, l24_15, acc0_24);
__m256 v0_14 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+14)*nb + i1+0]);
__m256i ws0_14 = _mm256_srli_epi32(w0, 28);
__m256i ws8_14 = _mm256_srli_epi32(w8, 28);
__m256i ws16_14 = _mm256_srli_epi32(w16, 28);
__m256i ws24_14 = _mm256_srli_epi32(w24, 28);
__m256i wsa0_14= _mm256_and_si256(ws0_14, mask);
__m256i wsa8_14= _mm256_and_si256(ws8_14, mask);
__m256i wsa16_14= _mm256_and_si256(ws16_14, mask);
__m256i wsa24_14= _mm256_and_si256(ws24_14, mask);
__m256 l0_14 = _mm256_cvtepi32_ps(wsa0_14);
__m256 l8_14 = _mm256_cvtepi32_ps(wsa8_14);
__m256 l16_14 = _mm256_cvtepi32_ps(wsa16_14);
__m256 l24_14 = _mm256_cvtepi32_ps(wsa24_14);
acc0_0 = _mm256_fmadd_ps(v0_14, l0_14, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_14, l8_14, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_14, l16_14, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_14, l24_14, acc0_24);
__m256 v0_13 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+13)*nb + i1+0]);
__m256i ws0_13 = _mm256_srli_epi32(w0, 26);
__m256i ws8_13 = _mm256_srli_epi32(w8, 26);
__m256i ws16_13 = _mm256_srli_epi32(w16, 26);
__m256i ws24_13 = _mm256_srli_epi32(w24, 26);
__m256i wsa0_13= _mm256_and_si256(ws0_13, mask);
__m256i wsa8_13= _mm256_and_si256(ws8_13, mask);
__m256i wsa16_13= _mm256_and_si256(ws16_13, mask);
__m256i wsa24_13= _mm256_and_si256(ws24_13, mask);
__m256 l0_13 = _mm256_cvtepi32_ps(wsa0_13);
__m256 l8_13 = _mm256_cvtepi32_ps(wsa8_13);
__m256 l16_13 = _mm256_cvtepi32_ps(wsa16_13);
__m256 l24_13 = _mm256_cvtepi32_ps(wsa24_13);
acc0_0 = _mm256_fmadd_ps(v0_13, l0_13, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_13, l8_13, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_13, l16_13, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_13, l24_13, acc0_24);
__m256 v0_12 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+12)*nb + i1+0]);
__m256i ws0_12 = _mm256_srli_epi32(w0, 24);
__m256i ws8_12 = _mm256_srli_epi32(w8, 24);
__m256i ws16_12 = _mm256_srli_epi32(w16, 24);
__m256i ws24_12 = _mm256_srli_epi32(w24, 24);
__m256i wsa0_12= _mm256_and_si256(ws0_12, mask);
__m256i wsa8_12= _mm256_and_si256(ws8_12, mask);
__m256i wsa16_12= _mm256_and_si256(ws16_12, mask);
__m256i wsa24_12= _mm256_and_si256(ws24_12, mask);
__m256 l0_12 = _mm256_cvtepi32_ps(wsa0_12);
__m256 l8_12 = _mm256_cvtepi32_ps(wsa8_12);
__m256 l16_12 = _mm256_cvtepi32_ps(wsa16_12);
__m256 l24_12 = _mm256_cvtepi32_ps(wsa24_12);
acc0_0 = _mm256_fmadd_ps(v0_12, l0_12, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_12, l8_12, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_12, l16_12, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_12, l24_12, acc0_24);
__m256 v0_11 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+11)*nb + i1+0]);
__m256i ws0_11 = _mm256_srli_epi32(w0, 22);
__m256i ws8_11 = _mm256_srli_epi32(w8, 22);
__m256i ws16_11 = _mm256_srli_epi32(w16, 22);
__m256i ws24_11 = _mm256_srli_epi32(w24, 22);
__m256i wsa0_11= _mm256_and_si256(ws0_11, mask);
__m256i wsa8_11= _mm256_and_si256(ws8_11, mask);
__m256i wsa16_11= _mm256_and_si256(ws16_11, mask);
__m256i wsa24_11= _mm256_and_si256(ws24_11, mask);
__m256 l0_11 = _mm256_cvtepi32_ps(wsa0_11);
__m256 l8_11 = _mm256_cvtepi32_ps(wsa8_11);
__m256 l16_11 = _mm256_cvtepi32_ps(wsa16_11);
__m256 l24_11 = _mm256_cvtepi32_ps(wsa24_11);
acc0_0 = _mm256_fmadd_ps(v0_11, l0_11, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_11, l8_11, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_11, l16_11, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_11, l24_11, acc0_24);
__m256 v0_10 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+10)*nb + i1+0]);
__m256i ws0_10 = _mm256_srli_epi32(w0, 20);
__m256i ws8_10 = _mm256_srli_epi32(w8, 20);
__m256i ws16_10 = _mm256_srli_epi32(w16, 20);
__m256i ws24_10 = _mm256_srli_epi32(w24, 20);
__m256i wsa0_10= _mm256_and_si256(ws0_10, mask);
__m256i wsa8_10= _mm256_and_si256(ws8_10, mask);
__m256i wsa16_10= _mm256_and_si256(ws16_10, mask);
__m256i wsa24_10= _mm256_and_si256(ws24_10, mask);
__m256 l0_10 = _mm256_cvtepi32_ps(wsa0_10);
__m256 l8_10 = _mm256_cvtepi32_ps(wsa8_10);
__m256 l16_10 = _mm256_cvtepi32_ps(wsa16_10);
__m256 l24_10 = _mm256_cvtepi32_ps(wsa24_10);
acc0_0 = _mm256_fmadd_ps(v0_10, l0_10, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_10, l8_10, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_10, l16_10, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_10, l24_10, acc0_24);
__m256 v0_9 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+9)*nb + i1+0]);
__m256i ws0_9 = _mm256_srli_epi32(w0, 18);
__m256i ws8_9 = _mm256_srli_epi32(w8, 18);
__m256i ws16_9 = _mm256_srli_epi32(w16, 18);
__m256i ws24_9 = _mm256_srli_epi32(w24, 18);
__m256i wsa0_9= _mm256_and_si256(ws0_9, mask);
__m256i wsa8_9= _mm256_and_si256(ws8_9, mask);
__m256i wsa16_9= _mm256_and_si256(ws16_9, mask);
__m256i wsa24_9= _mm256_and_si256(ws24_9, mask);
__m256 l0_9 = _mm256_cvtepi32_ps(wsa0_9);
__m256 l8_9 = _mm256_cvtepi32_ps(wsa8_9);
__m256 l16_9 = _mm256_cvtepi32_ps(wsa16_9);
__m256 l24_9 = _mm256_cvtepi32_ps(wsa24_9);
acc0_0 = _mm256_fmadd_ps(v0_9, l0_9, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_9, l8_9, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_9, l16_9, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_9, l24_9, acc0_24);
__m256 v0_8 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+8)*nb + i1+0]);
__m256i ws0_8 = _mm256_srli_epi32(w0, 16);
__m256i ws8_8 = _mm256_srli_epi32(w8, 16);
__m256i ws16_8 = _mm256_srli_epi32(w16, 16);
__m256i ws24_8 = _mm256_srli_epi32(w24, 16);
__m256i wsa0_8= _mm256_and_si256(ws0_8, mask);
__m256i wsa8_8= _mm256_and_si256(ws8_8, mask);
__m256i wsa16_8= _mm256_and_si256(ws16_8, mask);
__m256i wsa24_8= _mm256_and_si256(ws24_8, mask);
__m256 l0_8 = _mm256_cvtepi32_ps(wsa0_8);
__m256 l8_8 = _mm256_cvtepi32_ps(wsa8_8);
__m256 l16_8 = _mm256_cvtepi32_ps(wsa16_8);
__m256 l24_8 = _mm256_cvtepi32_ps(wsa24_8);
acc0_0 = _mm256_fmadd_ps(v0_8, l0_8, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_8, l8_8, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_8, l16_8, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_8, l24_8, acc0_24);
__m256 v0_7 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+7)*nb + i1+0]);
__m256i ws0_7 = _mm256_srli_epi32(w0, 14);
__m256i ws8_7 = _mm256_srli_epi32(w8, 14);
__m256i ws16_7 = _mm256_srli_epi32(w16, 14);
__m256i ws24_7 = _mm256_srli_epi32(w24, 14);
__m256i wsa0_7= _mm256_and_si256(ws0_7, mask);
__m256i wsa8_7= _mm256_and_si256(ws8_7, mask);
__m256i wsa16_7= _mm256_and_si256(ws16_7, mask);
__m256i wsa24_7= _mm256_and_si256(ws24_7, mask);
__m256 l0_7 = _mm256_cvtepi32_ps(wsa0_7);
__m256 l8_7 = _mm256_cvtepi32_ps(wsa8_7);
__m256 l16_7 = _mm256_cvtepi32_ps(wsa16_7);
__m256 l24_7 = _mm256_cvtepi32_ps(wsa24_7);
acc0_0 = _mm256_fmadd_ps(v0_7, l0_7, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_7, l8_7, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_7, l16_7, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_7, l24_7, acc0_24);
__m256 v0_6 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+6)*nb + i1+0]);
__m256i ws0_6 = _mm256_srli_epi32(w0, 12);
__m256i ws8_6 = _mm256_srli_epi32(w8, 12);
__m256i ws16_6 = _mm256_srli_epi32(w16, 12);
__m256i ws24_6 = _mm256_srli_epi32(w24, 12);
__m256i wsa0_6= _mm256_and_si256(ws0_6, mask);
__m256i wsa8_6= _mm256_and_si256(ws8_6, mask);
__m256i wsa16_6= _mm256_and_si256(ws16_6, mask);
__m256i wsa24_6= _mm256_and_si256(ws24_6, mask);
__m256 l0_6 = _mm256_cvtepi32_ps(wsa0_6);
__m256 l8_6 = _mm256_cvtepi32_ps(wsa8_6);
__m256 l16_6 = _mm256_cvtepi32_ps(wsa16_6);
__m256 l24_6 = _mm256_cvtepi32_ps(wsa24_6);
acc0_0 = _mm256_fmadd_ps(v0_6, l0_6, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_6, l8_6, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_6, l16_6, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_6, l24_6, acc0_24);
__m256 v0_5 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+5)*nb + i1+0]);
__m256i ws0_5 = _mm256_srli_epi32(w0, 10);
__m256i ws8_5 = _mm256_srli_epi32(w8, 10);
__m256i ws16_5 = _mm256_srli_epi32(w16, 10);
__m256i ws24_5 = _mm256_srli_epi32(w24, 10);
__m256i wsa0_5= _mm256_and_si256(ws0_5, mask);
__m256i wsa8_5= _mm256_and_si256(ws8_5, mask);
__m256i wsa16_5= _mm256_and_si256(ws16_5, mask);
__m256i wsa24_5= _mm256_and_si256(ws24_5, mask);
__m256 l0_5 = _mm256_cvtepi32_ps(wsa0_5);
__m256 l8_5 = _mm256_cvtepi32_ps(wsa8_5);
__m256 l16_5 = _mm256_cvtepi32_ps(wsa16_5);
__m256 l24_5 = _mm256_cvtepi32_ps(wsa24_5);
acc0_0 = _mm256_fmadd_ps(v0_5, l0_5, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_5, l8_5, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_5, l16_5, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_5, l24_5, acc0_24);
__m256 v0_4 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+4)*nb + i1+0]);
__m256i ws0_4 = _mm256_srli_epi32(w0, 8);
__m256i ws8_4 = _mm256_srli_epi32(w8, 8);
__m256i ws16_4 = _mm256_srli_epi32(w16, 8);
__m256i ws24_4 = _mm256_srli_epi32(w24, 8);
__m256i wsa0_4= _mm256_and_si256(ws0_4, mask);
__m256i wsa8_4= _mm256_and_si256(ws8_4, mask);
__m256i wsa16_4= _mm256_and_si256(ws16_4, mask);
__m256i wsa24_4= _mm256_and_si256(ws24_4, mask);
__m256 l0_4 = _mm256_cvtepi32_ps(wsa0_4);
__m256 l8_4 = _mm256_cvtepi32_ps(wsa8_4);
__m256 l16_4 = _mm256_cvtepi32_ps(wsa16_4);
__m256 l24_4 = _mm256_cvtepi32_ps(wsa24_4);
acc0_0 = _mm256_fmadd_ps(v0_4, l0_4, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_4, l8_4, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_4, l16_4, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_4, l24_4, acc0_24);
__m256 v0_3 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+3)*nb + i1+0]);
__m256i ws0_3 = _mm256_srli_epi32(w0, 6);
__m256i ws8_3 = _mm256_srli_epi32(w8, 6);
__m256i ws16_3 = _mm256_srli_epi32(w16, 6);
__m256i ws24_3 = _mm256_srli_epi32(w24, 6);
__m256i wsa0_3= _mm256_and_si256(ws0_3, mask);
__m256i wsa8_3= _mm256_and_si256(ws8_3, mask);
__m256i wsa16_3= _mm256_and_si256(ws16_3, mask);
__m256i wsa24_3= _mm256_and_si256(ws24_3, mask);
__m256 l0_3 = _mm256_cvtepi32_ps(wsa0_3);
__m256 l8_3 = _mm256_cvtepi32_ps(wsa8_3);
__m256 l16_3 = _mm256_cvtepi32_ps(wsa16_3);
__m256 l24_3 = _mm256_cvtepi32_ps(wsa24_3);
acc0_0 = _mm256_fmadd_ps(v0_3, l0_3, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_3, l8_3, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_3, l16_3, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_3, l24_3, acc0_24);
__m256 v0_2 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+2)*nb + i1+0]);
__m256i ws0_2 = _mm256_srli_epi32(w0, 4);
__m256i ws8_2 = _mm256_srli_epi32(w8, 4);
__m256i ws16_2 = _mm256_srli_epi32(w16, 4);
__m256i ws24_2 = _mm256_srli_epi32(w24, 4);
__m256i wsa0_2= _mm256_and_si256(ws0_2, mask);
__m256i wsa8_2= _mm256_and_si256(ws8_2, mask);
__m256i wsa16_2= _mm256_and_si256(ws16_2, mask);
__m256i wsa24_2= _mm256_and_si256(ws24_2, mask);
__m256 l0_2 = _mm256_cvtepi32_ps(wsa0_2);
__m256 l8_2 = _mm256_cvtepi32_ps(wsa8_2);
__m256 l16_2 = _mm256_cvtepi32_ps(wsa16_2);
__m256 l24_2 = _mm256_cvtepi32_ps(wsa24_2);
acc0_0 = _mm256_fmadd_ps(v0_2, l0_2, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_2, l8_2, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_2, l16_2, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_2, l24_2, acc0_24);
__m256 v0_1 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+1)*nb + i1+0]);
__m256i ws0_1 = _mm256_srli_epi32(w0, 2);
__m256i ws8_1 = _mm256_srli_epi32(w8, 2);
__m256i ws16_1 = _mm256_srli_epi32(w16, 2);
__m256i ws24_1 = _mm256_srli_epi32(w24, 2);
__m256i wsa0_1= _mm256_and_si256(ws0_1, mask);
__m256i wsa8_1= _mm256_and_si256(ws8_1, mask);
__m256i wsa16_1= _mm256_and_si256(ws16_1, mask);
__m256i wsa24_1= _mm256_and_si256(ws24_1, mask);
__m256 l0_1 = _mm256_cvtepi32_ps(wsa0_1);
__m256 l8_1 = _mm256_cvtepi32_ps(wsa8_1);
__m256 l16_1 = _mm256_cvtepi32_ps(wsa16_1);
__m256 l24_1 = _mm256_cvtepi32_ps(wsa24_1);
acc0_0 = _mm256_fmadd_ps(v0_1, l0_1, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_1, l8_1, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_1, l16_1, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_1, l24_1, acc0_24);
__m256 v0_0 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+0)*nb + i1+0]);
__m256i ws0_0 = _mm256_srli_epi32(w0, 0);
__m256i ws8_0 = _mm256_srli_epi32(w8, 0);
__m256i ws16_0 = _mm256_srli_epi32(w16, 0);
__m256i ws24_0 = _mm256_srli_epi32(w24, 0);
__m256i wsa0_0= _mm256_and_si256(ws0_0, mask);
__m256i wsa8_0= _mm256_and_si256(ws8_0, mask);
__m256i wsa16_0= _mm256_and_si256(ws16_0, mask);
__m256i wsa24_0= _mm256_and_si256(ws24_0, mask);
__m256 l0_0 = _mm256_cvtepi32_ps(wsa0_0);
__m256 l8_0 = _mm256_cvtepi32_ps(wsa8_0);
__m256 l16_0 = _mm256_cvtepi32_ps(wsa16_0);
__m256 l24_0 = _mm256_cvtepi32_ps(wsa24_0);
acc0_0 = _mm256_fmadd_ps(v0_0, l0_0, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_0, l8_0, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_0, l16_0, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_0, l24_0, acc0_24);
}
__m256 o0_0 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+0]);
__m256 o0_8 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+8]);
__m256 o0_16 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+16]);
__m256 o0_24 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+24]);
__m256 s0_0 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+0]);
__m256 s0_8 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+8]);
__m256 s0_16 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+16]);
__m256 s0_24 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+24]);
__m256 f0_0 = _mm256_fmadd_ps(acc0_0, s0_0, o0_0);
__m256 f0_8 = _mm256_fmadd_ps(acc0_8, s0_8, o0_8);
__m256 f0_16 = _mm256_fmadd_ps(acc0_16, s0_16, o0_16);
__m256 f0_24 = _mm256_fmadd_ps(acc0_24, s0_24, o0_24);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+0], f0_0);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+8], f0_8);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+16], f0_16);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+24], f0_24);
}
}
}
}
}
}
#pragma omp barrier
const int ngs = m/gs;
for (int i = 0; i < n; i++) {
for (int j = 0; j < tt; j+=32){
__m256 acc0 = _mm256_setzero_ps();
__m256 acc8 = _mm256_setzero_ps();
__m256 acc16 = _mm256_setzero_ps();
__m256 acc24 = _mm256_setzero_ps();
for (int i1 = 0; i1 < ngs; i1++){
__m256 r = _mm256_set1_ps(sums[i*ngs + i1]);
__m256 z0 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 0]);
__m256 z8 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 8]);
__m256 z16 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 16]);
__m256 z24 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 24]);
__m256 s0 = _mm256_loadu_ps(&scales[base_output + i1 * t + j + 0]);
__m256 s8 = _mm256_loadu_ps(&scales[base_output + i1 * t + j + 8]);
__m256 s16 = _mm256_loadu_ps(&scales[base_output + i1 * t + j + 16]);
__m256 s24 = _mm256_loadu_ps(&scales[base_output + i1 * t + j + 24]);
__m256 zs0 = _mm256_mul_ps(z0, s0);
__m256 zs8 = _mm256_mul_ps(z8, s8);
__m256 zs16 = _mm256_mul_ps(z16, s16);
__m256 zs24 = _mm256_mul_ps(z24, s24);
acc0 = _mm256_fmadd_ps(zs0, r, acc0);
acc8 = _mm256_fmadd_ps(zs8, r, acc8);
acc16 = _mm256_fmadd_ps(zs16, r, acc16);
acc24 = _mm256_fmadd_ps(zs24, r, acc24);
}
__m256 o0 = _mm256_loadu_ps(&output[i*t + base_output + j + 0]);
__m256 o8 = _mm256_loadu_ps(&output[i*t + base_output + j + 8]);
__m256 o16 = _mm256_loadu_ps(&output[i*t + base_output + j + 16]);
__m256 o24 = _mm256_loadu_ps(&output[i*t + base_output + j + 24]);
__m256 b0 = _mm256_loadu_ps(&bias[base_output + j + 0]);
__m256 b8 = _mm256_loadu_ps(&bias[base_output + j + 8]);
__m256 b16 = _mm256_loadu_ps(&bias[base_output + j + 16]);
__m256 b24 = _mm256_loadu_ps(&bias[base_output + j + 24]);
__m256 o10 = _mm256_sub_ps(o0, acc0);
__m256 o18 = _mm256_sub_ps(o8, acc8);
__m256 o116 = _mm256_sub_ps(o16, acc16);
__m256 o124 = _mm256_sub_ps(o24, acc24);
__m256 o20 = _mm256_add_ps(o10, b0);
__m256 o28 = _mm256_add_ps(o18, b8);
__m256 o216 = _mm256_add_ps(o116, b16);
__m256 o224 = _mm256_add_ps(o124, b24);
_mm256_storeu_ps(&output[i*t + base_output + j + 0], o20);
_mm256_storeu_ps(&output[i*t + base_output + j + 8], o28);
_mm256_storeu_ps(&output[i*t + base_output + j + 16], o216);
_mm256_storeu_ps(&output[i*t + base_output + j + 24], o224);
}
}
}
}
inline void forward2_gs_cpu(
torch::Tensor in, torch::Tensor weight, torch::Tensor out,
torch::Tensor bias, torch::Tensor scales, torch::Tensor zeros, torch::Tensor sums,
int N, int M, int T, int nb, int mb, int tb, int tt, int groupsize, int cutoff){
int*   W = weight.data_ptr<int>();
float* input = in.data_ptr<float>();
float* b   = bias.data_ptr<float>();
float* s   = scales.data_ptr<float>();
float* z   = zeros.data_ptr<float>();
float* r   = sums.data_ptr<float>();
float* O   = out.data_ptr<float>();

q2gemm_gs(input, W, s, z, b, r, O, N, M, T, nb, mb, tb, tt, groupsize, cutoff);
}
inline void pack2_qw_inner(int* A, int* B, const int N, const int M, const int nb, int mb, int cutoff){
// copy the full matrix A in blocked format into B
uint64_t idx = 0;
for(int j = 0, tid = 0; j < M; j+=mb, tid++){
for(int i = 0; i < N; i+=nb){
 for(int ii = i; ii < mymin(i+nb, N); ii++){
 for(int jj = j; jj < mymin(j+mb, M); jj++){
 B[idx] = A[ii*M+jj];
 idx++;
 }
 }
 }
}
}
inline void pack2_w_cpu(
torch::Tensor in, torch::Tensor out,
int N, int M, int nb, int mb, int cutoff){
int* input = in.data_ptr<int>();
int* O = out.data_ptr<int>();
  pack2_qw_inner(input, O, N, M, nb, mb, cutoff);
}
void unpack_zeros2_cpu(const int* zv, float* ov, int n, int m){
const __m256i ones = _mm256_set1_epi32(1);
const __m256i mask = _mm256_set1_epi32(3);
const __m256i shift0 = _mm256_set_epi32(30,28,26,24,22,20,18,16);
const __m256i shift1 = _mm256_set_epi32(14,12,10,8,6,4,2,0);
for(int i = 0; i < n; i++){
for (int j = 0; j < m; j+=16){
for (int k = 0; k < 16; k++){
ov[i*m + j+k] = (((zv[j/16] >> (2*k)) & 3)+1);
}
}
}
}
void unpack_zeros2(torch::Tensor zeros, torch::Tensor out, int N, int M){
int* Z = zeros.data_ptr<int>();
float* O = out.data_ptr<float>();
unpack_zeros2_cpu(Z, O, N, M);
}
inline
void q3gemm(const float* __restrict__ input, 
const int* __restrict__ W, 
const float* __restrict__ scales, 
const float* __restrict__ zeros, 
const float* __restrict__ bias, 
 const float* __restrict__ sums, 
 float* __restrict__ output,
const int n,
const int m,
const int t,
const int nb,
const int mb,
const int tb,
int ogtt,
const int cutoff){
#pragma omp parallel num_threads(12)
{
int tid;
const int mu = 16;
const int nu = 1;
const int tu = 16;
const int on = n / nb;
const int om = m / mb;
const __m256i mask = _mm256_set1_epi32(7);
const __m256i mask4 = _mm256_set1_epi32(4);
const __m256i mask6 = _mm256_set1_epi32(6);
tid = omp_get_thread_num();
int tt = ogtt;
if(tid >= cutoff){
tt -= tb;
}
const int base_output = tid >= cutoff ?
 (tid-cutoff)*tt + (tt+tb)*cutoff: 
 tid*tt;
const int base_W = tid >= cutoff ?
 ((tid-cutoff)*tt + (tt+tb)*cutoff)*m/32*3: 
 tid*tt*m/32*3;
for(int j = 0; j < tt; j+=tb){
for(int i = 0; i < on; i++) {
for(int k = 0; k < om; k++) {
for(int i1 = 0; i1 < nb; i1+=nu) {
int j1 = 0;
int jw = 0;
for(; j1 < tb-tu+1; j1+=tu, jw+=48){
__m256 acc0_0 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+0]);
__m256 acc0_8 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+8]);
for(int k1 = 0; k1 < mb; k1+=mu) {
for(int k2 = k1; k2 < k1+mu; k2+=32){
__m256i w0_0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+0]);
__m256i w1_0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+0+8]);
__m256i w2_0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+0+16]);
__m256i w0_8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+24]);
__m256i w1_8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+24+8]);
__m256i w2_8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+24+16]);
__m256 v0_0 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+0)*nb + i1+0]);
__m256 v0_1 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+1)*nb + i1+0]);
__m256i ws0_0 = _mm256_srli_epi32(w0_0, 0);
__m256i ws8_0 = _mm256_srli_epi32(w0_8, 0);
__m256i wsa0_0 = _mm256_and_si256(ws0_0, mask);
__m256i wsa8_0 = _mm256_and_si256(ws8_0, mask);
__m256 l0_0 = _mm256_cvtepi32_ps(wsa0_0);
__m256 l8_0 = _mm256_cvtepi32_ps(wsa8_0);
acc0_0 = _mm256_fmadd_ps(v0_0, l0_0, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_0, l8_0, acc0_8);
__m256i ws0_1 = _mm256_srli_epi32(w0_0, 3);
__m256i ws8_1 = _mm256_srli_epi32(w0_8, 3);
__m256i wsa0_1 = _mm256_and_si256(ws0_1, mask);
__m256i wsa8_1 = _mm256_and_si256(ws8_1, mask);
__m256 l0_1 = _mm256_cvtepi32_ps(wsa0_1);
__m256 l8_1 = _mm256_cvtepi32_ps(wsa8_1);
acc0_0 = _mm256_fmadd_ps(v0_1, l0_1, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_1, l8_1, acc0_8);
__m256 v0_2 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+2)*nb + i1+0]);
__m256 v0_3 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+3)*nb + i1+0]);
__m256i ws0_2 = _mm256_srli_epi32(w0_0, 6);
__m256i ws8_2 = _mm256_srli_epi32(w0_8, 6);
__m256i wsa0_2 = _mm256_and_si256(ws0_2, mask);
__m256i wsa8_2 = _mm256_and_si256(ws8_2, mask);
__m256 l0_2 = _mm256_cvtepi32_ps(wsa0_2);
__m256 l8_2 = _mm256_cvtepi32_ps(wsa8_2);
acc0_0 = _mm256_fmadd_ps(v0_2, l0_2, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_2, l8_2, acc0_8);
__m256i ws0_3 = _mm256_srli_epi32(w0_0, 9);
__m256i ws8_3 = _mm256_srli_epi32(w0_8, 9);
__m256i wsa0_3 = _mm256_and_si256(ws0_3, mask);
__m256i wsa8_3 = _mm256_and_si256(ws8_3, mask);
__m256 l0_3 = _mm256_cvtepi32_ps(wsa0_3);
__m256 l8_3 = _mm256_cvtepi32_ps(wsa8_3);
acc0_0 = _mm256_fmadd_ps(v0_3, l0_3, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_3, l8_3, acc0_8);
__m256 v0_4 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+4)*nb + i1+0]);
__m256 v0_5 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+5)*nb + i1+0]);
__m256i ws0_4 = _mm256_srli_epi32(w0_0, 12);
__m256i ws8_4 = _mm256_srli_epi32(w0_8, 12);
__m256i wsa0_4 = _mm256_and_si256(ws0_4, mask);
__m256i wsa8_4 = _mm256_and_si256(ws8_4, mask);
__m256 l0_4 = _mm256_cvtepi32_ps(wsa0_4);
__m256 l8_4 = _mm256_cvtepi32_ps(wsa8_4);
acc0_0 = _mm256_fmadd_ps(v0_4, l0_4, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_4, l8_4, acc0_8);
__m256i ws0_5 = _mm256_srli_epi32(w0_0, 15);
__m256i ws8_5 = _mm256_srli_epi32(w0_8, 15);
__m256i wsa0_5 = _mm256_and_si256(ws0_5, mask);
__m256i wsa8_5 = _mm256_and_si256(ws8_5, mask);
__m256 l0_5 = _mm256_cvtepi32_ps(wsa0_5);
__m256 l8_5 = _mm256_cvtepi32_ps(wsa8_5);
acc0_0 = _mm256_fmadd_ps(v0_5, l0_5, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_5, l8_5, acc0_8);
__m256 v0_6 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+6)*nb + i1+0]);
__m256 v0_7 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+7)*nb + i1+0]);
__m256i ws0_6 = _mm256_srli_epi32(w0_0, 18);
__m256i ws8_6 = _mm256_srli_epi32(w0_8, 18);
__m256i wsa0_6 = _mm256_and_si256(ws0_6, mask);
__m256i wsa8_6 = _mm256_and_si256(ws8_6, mask);
__m256 l0_6 = _mm256_cvtepi32_ps(wsa0_6);
__m256 l8_6 = _mm256_cvtepi32_ps(wsa8_6);
acc0_0 = _mm256_fmadd_ps(v0_6, l0_6, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_6, l8_6, acc0_8);
__m256i ws0_7 = _mm256_srli_epi32(w0_0, 21);
__m256i ws8_7 = _mm256_srli_epi32(w0_8, 21);
__m256i wsa0_7 = _mm256_and_si256(ws0_7, mask);
__m256i wsa8_7 = _mm256_and_si256(ws8_7, mask);
__m256 l0_7 = _mm256_cvtepi32_ps(wsa0_7);
__m256 l8_7 = _mm256_cvtepi32_ps(wsa8_7);
acc0_0 = _mm256_fmadd_ps(v0_7, l0_7, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_7, l8_7, acc0_8);
__m256 v0_8 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+8)*nb + i1+0]);
__m256 v0_9 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+9)*nb + i1+0]);
__m256i ws0_8 = _mm256_srli_epi32(w0_0, 24);
__m256i ws8_8 = _mm256_srli_epi32(w0_8, 24);
__m256i wsa0_8 = _mm256_and_si256(ws0_8, mask);
__m256i wsa8_8 = _mm256_and_si256(ws8_8, mask);
__m256 l0_8 = _mm256_cvtepi32_ps(wsa0_8);
__m256 l8_8 = _mm256_cvtepi32_ps(wsa8_8);
acc0_0 = _mm256_fmadd_ps(v0_8, l0_8, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_8, l8_8, acc0_8);
__m256i ws0_9 = _mm256_srli_epi32(w0_0, 27);
__m256i ws8_9 = _mm256_srli_epi32(w0_8, 27);
__m256i wsa0_9 = _mm256_and_si256(ws0_9, mask);
__m256i wsa8_9 = _mm256_and_si256(ws8_9, mask);
__m256 l0_9 = _mm256_cvtepi32_ps(wsa0_9);
__m256 l8_9 = _mm256_cvtepi32_ps(wsa8_9);
acc0_0 = _mm256_fmadd_ps(v0_9, l0_9, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_9, l8_9, acc0_8);
__m256 v0_10 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+10)*nb + i1+0]);
__m256i ws0_10 = _mm256_srli_epi32(w0_0, 30);
__m256i temp0_0 = _mm256_slli_epi32(w1_0, 2);
temp0_0 = _mm256_and_si256(temp0_0, mask);
ws0_10 = _mm256_or_si256(ws0_10, temp0_0);
__m256i wsa0_10 = _mm256_and_si256(ws0_10, mask);
__m256 l0_10 = _mm256_cvtepi32_ps(wsa0_10);
acc0_0 = _mm256_fmadd_ps(v0_10, l0_10, acc0_0);
__m256i ws8_10 = _mm256_srli_epi32(w0_8, 30);
__m256i temp0_8 = _mm256_slli_epi32(w1_8, 2);
temp0_8 = _mm256_and_si256(temp0_8, mask);
ws8_10 = _mm256_or_si256(ws8_10, temp0_8);
__m256i wsa8_10 = _mm256_and_si256(ws8_10, mask);
__m256 l8_10 = _mm256_cvtepi32_ps(wsa8_10);
acc0_8 = _mm256_fmadd_ps(v0_10, l8_10, acc0_8);
__m256 v0_11 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+11)*nb + i1+0]);
__m256 v0_12 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+12)*nb + i1+0]);
__m256i ws0_11 = _mm256_srli_epi32(w1_0, 1);
__m256i ws8_11 = _mm256_srli_epi32(w1_8, 1);
__m256i wsa0_11 = _mm256_and_si256(ws0_11, mask);
__m256i wsa8_11 = _mm256_and_si256(ws8_11, mask);
__m256 l0_11 = _mm256_cvtepi32_ps(wsa0_11);
__m256 l8_11 = _mm256_cvtepi32_ps(wsa8_11);
acc0_0 = _mm256_fmadd_ps(v0_11, l0_11, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_11, l8_11, acc0_8);
__m256i ws0_12 = _mm256_srli_epi32(w1_0, 4);
__m256i ws8_12 = _mm256_srli_epi32(w1_8, 4);
__m256i wsa0_12 = _mm256_and_si256(ws0_12, mask);
__m256i wsa8_12 = _mm256_and_si256(ws8_12, mask);
__m256 l0_12 = _mm256_cvtepi32_ps(wsa0_12);
__m256 l8_12 = _mm256_cvtepi32_ps(wsa8_12);
acc0_0 = _mm256_fmadd_ps(v0_12, l0_12, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_12, l8_12, acc0_8);
__m256 v0_13 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+13)*nb + i1+0]);
__m256 v0_14 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+14)*nb + i1+0]);
__m256i ws0_13 = _mm256_srli_epi32(w1_0, 7);
__m256i ws8_13 = _mm256_srli_epi32(w1_8, 7);
__m256i wsa0_13 = _mm256_and_si256(ws0_13, mask);
__m256i wsa8_13 = _mm256_and_si256(ws8_13, mask);
__m256 l0_13 = _mm256_cvtepi32_ps(wsa0_13);
__m256 l8_13 = _mm256_cvtepi32_ps(wsa8_13);
acc0_0 = _mm256_fmadd_ps(v0_13, l0_13, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_13, l8_13, acc0_8);
__m256i ws0_14 = _mm256_srli_epi32(w1_0, 10);
__m256i ws8_14 = _mm256_srli_epi32(w1_8, 10);
__m256i wsa0_14 = _mm256_and_si256(ws0_14, mask);
__m256i wsa8_14 = _mm256_and_si256(ws8_14, mask);
__m256 l0_14 = _mm256_cvtepi32_ps(wsa0_14);
__m256 l8_14 = _mm256_cvtepi32_ps(wsa8_14);
acc0_0 = _mm256_fmadd_ps(v0_14, l0_14, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_14, l8_14, acc0_8);
__m256 v0_15 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+15)*nb + i1+0]);
__m256 v0_16 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+16)*nb + i1+0]);
__m256i ws0_15 = _mm256_srli_epi32(w1_0, 13);
__m256i ws8_15 = _mm256_srli_epi32(w1_8, 13);
__m256i wsa0_15 = _mm256_and_si256(ws0_15, mask);
__m256i wsa8_15 = _mm256_and_si256(ws8_15, mask);
__m256 l0_15 = _mm256_cvtepi32_ps(wsa0_15);
__m256 l8_15 = _mm256_cvtepi32_ps(wsa8_15);
acc0_0 = _mm256_fmadd_ps(v0_15, l0_15, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_15, l8_15, acc0_8);
__m256i ws0_16 = _mm256_srli_epi32(w1_0, 16);
__m256i ws8_16 = _mm256_srli_epi32(w1_8, 16);
__m256i wsa0_16 = _mm256_and_si256(ws0_16, mask);
__m256i wsa8_16 = _mm256_and_si256(ws8_16, mask);
__m256 l0_16 = _mm256_cvtepi32_ps(wsa0_16);
__m256 l8_16 = _mm256_cvtepi32_ps(wsa8_16);
acc0_0 = _mm256_fmadd_ps(v0_16, l0_16, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_16, l8_16, acc0_8);
__m256 v0_17 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+17)*nb + i1+0]);
__m256 v0_18 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+18)*nb + i1+0]);
__m256i ws0_17 = _mm256_srli_epi32(w1_0, 19);
__m256i ws8_17 = _mm256_srli_epi32(w1_8, 19);
__m256i wsa0_17 = _mm256_and_si256(ws0_17, mask);
__m256i wsa8_17 = _mm256_and_si256(ws8_17, mask);
__m256 l0_17 = _mm256_cvtepi32_ps(wsa0_17);
__m256 l8_17 = _mm256_cvtepi32_ps(wsa8_17);
acc0_0 = _mm256_fmadd_ps(v0_17, l0_17, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_17, l8_17, acc0_8);
__m256i ws0_18 = _mm256_srli_epi32(w1_0, 22);
__m256i ws8_18 = _mm256_srli_epi32(w1_8, 22);
__m256i wsa0_18 = _mm256_and_si256(ws0_18, mask);
__m256i wsa8_18 = _mm256_and_si256(ws8_18, mask);
__m256 l0_18 = _mm256_cvtepi32_ps(wsa0_18);
__m256 l8_18 = _mm256_cvtepi32_ps(wsa8_18);
acc0_0 = _mm256_fmadd_ps(v0_18, l0_18, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_18, l8_18, acc0_8);
__m256 v0_19 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+19)*nb + i1+0]);
__m256 v0_20 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+20)*nb + i1+0]);
__m256i ws0_19 = _mm256_srli_epi32(w1_0, 25);
__m256i ws8_19 = _mm256_srli_epi32(w1_8, 25);
__m256i wsa0_19 = _mm256_and_si256(ws0_19, mask);
__m256i wsa8_19 = _mm256_and_si256(ws8_19, mask);
__m256 l0_19 = _mm256_cvtepi32_ps(wsa0_19);
__m256 l8_19 = _mm256_cvtepi32_ps(wsa8_19);
acc0_0 = _mm256_fmadd_ps(v0_19, l0_19, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_19, l8_19, acc0_8);
__m256i ws0_20 = _mm256_srli_epi32(w1_0, 28);
__m256i ws8_20 = _mm256_srli_epi32(w1_8, 28);
__m256i wsa0_20 = _mm256_and_si256(ws0_20, mask);
__m256i wsa8_20 = _mm256_and_si256(ws8_20, mask);
__m256 l0_20 = _mm256_cvtepi32_ps(wsa0_20);
__m256 l8_20 = _mm256_cvtepi32_ps(wsa8_20);
acc0_0 = _mm256_fmadd_ps(v0_20, l0_20, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_20, l8_20, acc0_8);
__m256 v0_21 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+21)*nb + i1+0]);
__m256i ws0_21 = _mm256_srli_epi32(w1_0, 31);
__m256i temp1_0 = _mm256_slli_epi32(w2_0, 1);
temp1_0 = _mm256_and_si256(temp1_0, mask);
ws0_21 = _mm256_or_si256(ws0_21, temp1_0);
__m256i wsa0_21 = _mm256_and_si256(ws0_21, mask);
__m256 l0_21 = _mm256_cvtepi32_ps(wsa0_21);
acc0_0 = _mm256_fmadd_ps(v0_21, l0_21, acc0_0);
__m256i ws8_21 = _mm256_srli_epi32(w1_8, 31);
__m256i temp1_8 = _mm256_slli_epi32(w2_8, 1);
temp1_8 = _mm256_and_si256(temp1_8, mask);
ws8_21 = _mm256_or_si256(ws8_21, temp1_8);
__m256i wsa8_21 = _mm256_and_si256(ws8_21, mask);
__m256 l8_21 = _mm256_cvtepi32_ps(wsa8_21);
acc0_8 = _mm256_fmadd_ps(v0_21, l8_21, acc0_8);
__m256 v0_22 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+22)*nb + i1+0]);
__m256 v0_23 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+23)*nb + i1+0]);
__m256i ws0_22 = _mm256_srli_epi32(w2_0, 2);
__m256i ws8_22 = _mm256_srli_epi32(w2_8, 2);
__m256i wsa0_22 = _mm256_and_si256(ws0_22, mask);
__m256i wsa8_22 = _mm256_and_si256(ws8_22, mask);
__m256 l0_22 = _mm256_cvtepi32_ps(wsa0_22);
__m256 l8_22 = _mm256_cvtepi32_ps(wsa8_22);
acc0_0 = _mm256_fmadd_ps(v0_22, l0_22, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_22, l8_22, acc0_8);
__m256i ws0_23 = _mm256_srli_epi32(w2_0, 5);
__m256i ws8_23 = _mm256_srli_epi32(w2_8, 5);
__m256i wsa0_23 = _mm256_and_si256(ws0_23, mask);
__m256i wsa8_23 = _mm256_and_si256(ws8_23, mask);
__m256 l0_23 = _mm256_cvtepi32_ps(wsa0_23);
__m256 l8_23 = _mm256_cvtepi32_ps(wsa8_23);
acc0_0 = _mm256_fmadd_ps(v0_23, l0_23, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_23, l8_23, acc0_8);
__m256 v0_24 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+24)*nb + i1+0]);
__m256 v0_25 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+25)*nb + i1+0]);
__m256i ws0_24 = _mm256_srli_epi32(w2_0, 8);
__m256i ws8_24 = _mm256_srli_epi32(w2_8, 8);
__m256i wsa0_24 = _mm256_and_si256(ws0_24, mask);
__m256i wsa8_24 = _mm256_and_si256(ws8_24, mask);
__m256 l0_24 = _mm256_cvtepi32_ps(wsa0_24);
__m256 l8_24 = _mm256_cvtepi32_ps(wsa8_24);
acc0_0 = _mm256_fmadd_ps(v0_24, l0_24, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_24, l8_24, acc0_8);
__m256i ws0_25 = _mm256_srli_epi32(w2_0, 11);
__m256i ws8_25 = _mm256_srli_epi32(w2_8, 11);
__m256i wsa0_25 = _mm256_and_si256(ws0_25, mask);
__m256i wsa8_25 = _mm256_and_si256(ws8_25, mask);
__m256 l0_25 = _mm256_cvtepi32_ps(wsa0_25);
__m256 l8_25 = _mm256_cvtepi32_ps(wsa8_25);
acc0_0 = _mm256_fmadd_ps(v0_25, l0_25, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_25, l8_25, acc0_8);
__m256 v0_26 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+26)*nb + i1+0]);
__m256 v0_27 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+27)*nb + i1+0]);
__m256i ws0_26 = _mm256_srli_epi32(w2_0, 14);
__m256i ws8_26 = _mm256_srli_epi32(w2_8, 14);
__m256i wsa0_26 = _mm256_and_si256(ws0_26, mask);
__m256i wsa8_26 = _mm256_and_si256(ws8_26, mask);
__m256 l0_26 = _mm256_cvtepi32_ps(wsa0_26);
__m256 l8_26 = _mm256_cvtepi32_ps(wsa8_26);
acc0_0 = _mm256_fmadd_ps(v0_26, l0_26, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_26, l8_26, acc0_8);
__m256i ws0_27 = _mm256_srli_epi32(w2_0, 17);
__m256i ws8_27 = _mm256_srli_epi32(w2_8, 17);
__m256i wsa0_27 = _mm256_and_si256(ws0_27, mask);
__m256i wsa8_27 = _mm256_and_si256(ws8_27, mask);
__m256 l0_27 = _mm256_cvtepi32_ps(wsa0_27);
__m256 l8_27 = _mm256_cvtepi32_ps(wsa8_27);
acc0_0 = _mm256_fmadd_ps(v0_27, l0_27, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_27, l8_27, acc0_8);
__m256 v0_28 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+28)*nb + i1+0]);
__m256 v0_29 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+29)*nb + i1+0]);
__m256i ws0_28 = _mm256_srli_epi32(w2_0, 20);
__m256i ws8_28 = _mm256_srli_epi32(w2_8, 20);
__m256i wsa0_28 = _mm256_and_si256(ws0_28, mask);
__m256i wsa8_28 = _mm256_and_si256(ws8_28, mask);
__m256 l0_28 = _mm256_cvtepi32_ps(wsa0_28);
__m256 l8_28 = _mm256_cvtepi32_ps(wsa8_28);
acc0_0 = _mm256_fmadd_ps(v0_28, l0_28, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_28, l8_28, acc0_8);
__m256i ws0_29 = _mm256_srli_epi32(w2_0, 23);
__m256i ws8_29 = _mm256_srli_epi32(w2_8, 23);
__m256i wsa0_29 = _mm256_and_si256(ws0_29, mask);
__m256i wsa8_29 = _mm256_and_si256(ws8_29, mask);
__m256 l0_29 = _mm256_cvtepi32_ps(wsa0_29);
__m256 l8_29 = _mm256_cvtepi32_ps(wsa8_29);
acc0_0 = _mm256_fmadd_ps(v0_29, l0_29, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_29, l8_29, acc0_8);
__m256 v0_30 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+30)*nb + i1+0]);
__m256 v0_31 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+31)*nb + i1+0]);
__m256i ws0_30 = _mm256_srli_epi32(w2_0, 26);
__m256i ws8_30 = _mm256_srli_epi32(w2_8, 26);
__m256i wsa0_30 = _mm256_and_si256(ws0_30, mask);
__m256i wsa8_30 = _mm256_and_si256(ws8_30, mask);
__m256 l0_30 = _mm256_cvtepi32_ps(wsa0_30);
__m256 l8_30 = _mm256_cvtepi32_ps(wsa8_30);
acc0_0 = _mm256_fmadd_ps(v0_30, l0_30, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_30, l8_30, acc0_8);
__m256i ws0_31 = _mm256_srli_epi32(w2_0, 29);
__m256i ws8_31 = _mm256_srli_epi32(w2_8, 29);
__m256i wsa0_31 = _mm256_and_si256(ws0_31, mask);
__m256i wsa8_31 = _mm256_and_si256(ws8_31, mask);
__m256 l0_31 = _mm256_cvtepi32_ps(wsa0_31);
__m256 l8_31 = _mm256_cvtepi32_ps(wsa8_31);
acc0_0 = _mm256_fmadd_ps(v0_31, l0_31, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_31, l8_31, acc0_8);
}
}
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+0], acc0_0);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+8], acc0_8);
}
}
}
}
}
#pragma omp barrier
for (int i = 0; i < n; i++) {
__m256 r = _mm256_set1_ps(sums[i]);
for (int j = 0; j < tt; j+=16){
__m256 o0 = _mm256_loadu_ps(&output[i*t + base_output + j + 0]);
__m256 o8 = _mm256_loadu_ps(&output[i*t + base_output + j + 8]);
__m256 z0 = _mm256_loadu_ps(&zeros[base_output + j + 0]);
__m256 z8 = _mm256_loadu_ps(&zeros[base_output + j + 8]);
__m256 b0 = _mm256_loadu_ps(&bias[base_output + j + 0]);
__m256 b8 = _mm256_loadu_ps(&bias[base_output + j + 8]);
__m256 s0 = _mm256_loadu_ps(&scales[base_output + j + 0]);
__m256 s8 = _mm256_loadu_ps(&scales[base_output + j + 8]);
__m256 os0 = _mm256_mul_ps(o0, s0);
__m256 os8 = _mm256_mul_ps(o8, s8);
__m256 zr0 = _mm256_fnmadd_ps(z0, r, os0);
__m256 zr8 = _mm256_fnmadd_ps(z8, r, os8);
__m256 o20 = _mm256_add_ps(zr0, b0);
__m256 o28 = _mm256_add_ps(zr8, b8);
_mm256_storeu_ps(&output[i*t + base_output + j + 0], o20);
_mm256_storeu_ps(&output[i*t + base_output + j + 8], o28);
}
}
}
}
inline void forward3_cpu(
torch::Tensor in, torch::Tensor weight, torch::Tensor out,
torch::Tensor bias, torch::Tensor scales, torch::Tensor zeros, torch::Tensor sums,
int N, int M, int T, int nb, int mb, int tb, int tt, int cutoff){
int*   W = weight.data_ptr<int>();
float* input = in.data_ptr<float>();
float* b   = bias.data_ptr<float>();
float* s   = scales.data_ptr<float>();
float* z   = zeros.data_ptr<float>();
float* r   = sums.data_ptr<float>();
float* O   = out.data_ptr<float>();

q3gemm(input, W, s, z, b, r, O, N, M, T, nb, mb, tb, tt, cutoff);
}
inline
void q3gemm_gs(const float* __restrict__ input, 
const int* __restrict__ W, 
const float* __restrict__ scales, 
const float* __restrict__ zeros, 
const float* __restrict__ bias, 
 const float* __restrict__ sums, 
 float* __restrict__ output,
const int n,
const int m,
const int t,
const int nb,
const int mb,
const int tb,
int ogtt,
const int gs,
const int cutoff){
#pragma omp parallel num_threads(12)
{
int tid;
const int mu = 16;
const int nu = 1;
const int tu = 32;
const int on = n / nb;
const int om = m / mb;
const __m256i mask = _mm256_set1_epi32(7);
const __m256i mask4 = _mm256_set1_epi32(4);
const __m256i mask6 = _mm256_set1_epi32(6);
tid = omp_get_thread_num();
int tt = ogtt;
if(tid >= cutoff){
tt -= tb;
}
const int base_output = tid >= cutoff ?
 (tid-cutoff)*tt + (tt+tb)*cutoff: 
 tid*tt;
const int base_W = tid >= cutoff ?
 ((tid-cutoff)*tt + (tt+tb)*cutoff)*m/32*3: 
 tid*tt*m/32*3;
for(int j = 0; j < tt; j+=tb){
for(int i = 0; i < on; i++) {
for(int k = 0; k < om; k++) {
for(int i1 = 0; i1 < nb; i1+=nu) {
int j1 = 0;
int jw = 0;
for(; j1 < tb-tu+1; j1+=tu, jw+=96){
for(int k1 = 0; k1 < mb; k1+=gs) {
__m256 acc0_0 = _mm256_setzero_ps();
__m256 acc0_8 = _mm256_setzero_ps();
__m256 acc0_16 = _mm256_setzero_ps();
__m256 acc0_24 = _mm256_setzero_ps();
for(int k2 = k1; k2 < k1+gs; k2+=32)
{
__m256i w0_0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+0]);
__m256i w1_0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+0+8]);
__m256i w2_0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+0+16]);
__m256i w0_8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+24]);
__m256i w1_8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+24+8]);
__m256i w2_8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+24+16]);
__m256i w0_16 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+48]);
__m256i w1_16 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+48+8]);
__m256i w2_16 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+48+16]);
__m256i w0_24 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+72]);
__m256i w1_24 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+72+8]);
__m256i w2_24 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/32*3 + k*mb*tb/32*3 + k2*tb/32*3 + jw+72+16]);
__m256 v0_0 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+0)*nb + i1+0]);
__m256 v0_1 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+1)*nb + i1+0]);
__m256i ws0_0 = _mm256_srli_epi32(w0_0, 0);
__m256i ws8_0 = _mm256_srli_epi32(w0_8, 0);
__m256i ws16_0 = _mm256_srli_epi32(w0_16, 0);
__m256i ws24_0 = _mm256_srli_epi32(w0_24, 0);
__m256i wsa0_0 = _mm256_and_si256(ws0_0, mask);
__m256i wsa8_0 = _mm256_and_si256(ws8_0, mask);
__m256i wsa16_0 = _mm256_and_si256(ws16_0, mask);
__m256i wsa24_0 = _mm256_and_si256(ws24_0, mask);
__m256 l0_0 = _mm256_cvtepi32_ps(wsa0_0);
__m256 l8_0 = _mm256_cvtepi32_ps(wsa8_0);
__m256 l16_0 = _mm256_cvtepi32_ps(wsa16_0);
__m256 l24_0 = _mm256_cvtepi32_ps(wsa24_0);
acc0_0 = _mm256_fmadd_ps(v0_0, l0_0, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_0, l8_0, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_0, l16_0, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_0, l24_0, acc0_24);
__m256i ws0_1 = _mm256_srli_epi32(w0_0, 3);
__m256i ws8_1 = _mm256_srli_epi32(w0_8, 3);
__m256i ws16_1 = _mm256_srli_epi32(w0_16, 3);
__m256i ws24_1 = _mm256_srli_epi32(w0_24, 3);
__m256i wsa0_1 = _mm256_and_si256(ws0_1, mask);
__m256i wsa8_1 = _mm256_and_si256(ws8_1, mask);
__m256i wsa16_1 = _mm256_and_si256(ws16_1, mask);
__m256i wsa24_1 = _mm256_and_si256(ws24_1, mask);
__m256 l0_1 = _mm256_cvtepi32_ps(wsa0_1);
__m256 l8_1 = _mm256_cvtepi32_ps(wsa8_1);
__m256 l16_1 = _mm256_cvtepi32_ps(wsa16_1);
__m256 l24_1 = _mm256_cvtepi32_ps(wsa24_1);
acc0_0 = _mm256_fmadd_ps(v0_1, l0_1, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_1, l8_1, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_1, l16_1, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_1, l24_1, acc0_24);
__m256 v0_2 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+2)*nb + i1+0]);
__m256 v0_3 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+3)*nb + i1+0]);
__m256i ws0_2 = _mm256_srli_epi32(w0_0, 6);
__m256i ws8_2 = _mm256_srli_epi32(w0_8, 6);
__m256i ws16_2 = _mm256_srli_epi32(w0_16, 6);
__m256i ws24_2 = _mm256_srli_epi32(w0_24, 6);
__m256i wsa0_2 = _mm256_and_si256(ws0_2, mask);
__m256i wsa8_2 = _mm256_and_si256(ws8_2, mask);
__m256i wsa16_2 = _mm256_and_si256(ws16_2, mask);
__m256i wsa24_2 = _mm256_and_si256(ws24_2, mask);
__m256 l0_2 = _mm256_cvtepi32_ps(wsa0_2);
__m256 l8_2 = _mm256_cvtepi32_ps(wsa8_2);
__m256 l16_2 = _mm256_cvtepi32_ps(wsa16_2);
__m256 l24_2 = _mm256_cvtepi32_ps(wsa24_2);
acc0_0 = _mm256_fmadd_ps(v0_2, l0_2, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_2, l8_2, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_2, l16_2, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_2, l24_2, acc0_24);
__m256i ws0_3 = _mm256_srli_epi32(w0_0, 9);
__m256i ws8_3 = _mm256_srli_epi32(w0_8, 9);
__m256i ws16_3 = _mm256_srli_epi32(w0_16, 9);
__m256i ws24_3 = _mm256_srli_epi32(w0_24, 9);
__m256i wsa0_3 = _mm256_and_si256(ws0_3, mask);
__m256i wsa8_3 = _mm256_and_si256(ws8_3, mask);
__m256i wsa16_3 = _mm256_and_si256(ws16_3, mask);
__m256i wsa24_3 = _mm256_and_si256(ws24_3, mask);
__m256 l0_3 = _mm256_cvtepi32_ps(wsa0_3);
__m256 l8_3 = _mm256_cvtepi32_ps(wsa8_3);
__m256 l16_3 = _mm256_cvtepi32_ps(wsa16_3);
__m256 l24_3 = _mm256_cvtepi32_ps(wsa24_3);
acc0_0 = _mm256_fmadd_ps(v0_3, l0_3, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_3, l8_3, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_3, l16_3, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_3, l24_3, acc0_24);
__m256 v0_4 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+4)*nb + i1+0]);
__m256 v0_5 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+5)*nb + i1+0]);
__m256i ws0_4 = _mm256_srli_epi32(w0_0, 12);
__m256i ws8_4 = _mm256_srli_epi32(w0_8, 12);
__m256i ws16_4 = _mm256_srli_epi32(w0_16, 12);
__m256i ws24_4 = _mm256_srli_epi32(w0_24, 12);
__m256i wsa0_4 = _mm256_and_si256(ws0_4, mask);
__m256i wsa8_4 = _mm256_and_si256(ws8_4, mask);
__m256i wsa16_4 = _mm256_and_si256(ws16_4, mask);
__m256i wsa24_4 = _mm256_and_si256(ws24_4, mask);
__m256 l0_4 = _mm256_cvtepi32_ps(wsa0_4);
__m256 l8_4 = _mm256_cvtepi32_ps(wsa8_4);
__m256 l16_4 = _mm256_cvtepi32_ps(wsa16_4);
__m256 l24_4 = _mm256_cvtepi32_ps(wsa24_4);
acc0_0 = _mm256_fmadd_ps(v0_4, l0_4, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_4, l8_4, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_4, l16_4, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_4, l24_4, acc0_24);
__m256i ws0_5 = _mm256_srli_epi32(w0_0, 15);
__m256i ws8_5 = _mm256_srli_epi32(w0_8, 15);
__m256i ws16_5 = _mm256_srli_epi32(w0_16, 15);
__m256i ws24_5 = _mm256_srli_epi32(w0_24, 15);
__m256i wsa0_5 = _mm256_and_si256(ws0_5, mask);
__m256i wsa8_5 = _mm256_and_si256(ws8_5, mask);
__m256i wsa16_5 = _mm256_and_si256(ws16_5, mask);
__m256i wsa24_5 = _mm256_and_si256(ws24_5, mask);
__m256 l0_5 = _mm256_cvtepi32_ps(wsa0_5);
__m256 l8_5 = _mm256_cvtepi32_ps(wsa8_5);
__m256 l16_5 = _mm256_cvtepi32_ps(wsa16_5);
__m256 l24_5 = _mm256_cvtepi32_ps(wsa24_5);
acc0_0 = _mm256_fmadd_ps(v0_5, l0_5, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_5, l8_5, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_5, l16_5, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_5, l24_5, acc0_24);
__m256 v0_6 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+6)*nb + i1+0]);
__m256 v0_7 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+7)*nb + i1+0]);
__m256i ws0_6 = _mm256_srli_epi32(w0_0, 18);
__m256i ws8_6 = _mm256_srli_epi32(w0_8, 18);
__m256i ws16_6 = _mm256_srli_epi32(w0_16, 18);
__m256i ws24_6 = _mm256_srli_epi32(w0_24, 18);
__m256i wsa0_6 = _mm256_and_si256(ws0_6, mask);
__m256i wsa8_6 = _mm256_and_si256(ws8_6, mask);
__m256i wsa16_6 = _mm256_and_si256(ws16_6, mask);
__m256i wsa24_6 = _mm256_and_si256(ws24_6, mask);
__m256 l0_6 = _mm256_cvtepi32_ps(wsa0_6);
__m256 l8_6 = _mm256_cvtepi32_ps(wsa8_6);
__m256 l16_6 = _mm256_cvtepi32_ps(wsa16_6);
__m256 l24_6 = _mm256_cvtepi32_ps(wsa24_6);
acc0_0 = _mm256_fmadd_ps(v0_6, l0_6, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_6, l8_6, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_6, l16_6, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_6, l24_6, acc0_24);
__m256i ws0_7 = _mm256_srli_epi32(w0_0, 21);
__m256i ws8_7 = _mm256_srli_epi32(w0_8, 21);
__m256i ws16_7 = _mm256_srli_epi32(w0_16, 21);
__m256i ws24_7 = _mm256_srli_epi32(w0_24, 21);
__m256i wsa0_7 = _mm256_and_si256(ws0_7, mask);
__m256i wsa8_7 = _mm256_and_si256(ws8_7, mask);
__m256i wsa16_7 = _mm256_and_si256(ws16_7, mask);
__m256i wsa24_7 = _mm256_and_si256(ws24_7, mask);
__m256 l0_7 = _mm256_cvtepi32_ps(wsa0_7);
__m256 l8_7 = _mm256_cvtepi32_ps(wsa8_7);
__m256 l16_7 = _mm256_cvtepi32_ps(wsa16_7);
__m256 l24_7 = _mm256_cvtepi32_ps(wsa24_7);
acc0_0 = _mm256_fmadd_ps(v0_7, l0_7, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_7, l8_7, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_7, l16_7, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_7, l24_7, acc0_24);
__m256 v0_8 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+8)*nb + i1+0]);
__m256 v0_9 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+9)*nb + i1+0]);
__m256i ws0_8 = _mm256_srli_epi32(w0_0, 24);
__m256i ws8_8 = _mm256_srli_epi32(w0_8, 24);
__m256i ws16_8 = _mm256_srli_epi32(w0_16, 24);
__m256i ws24_8 = _mm256_srli_epi32(w0_24, 24);
__m256i wsa0_8 = _mm256_and_si256(ws0_8, mask);
__m256i wsa8_8 = _mm256_and_si256(ws8_8, mask);
__m256i wsa16_8 = _mm256_and_si256(ws16_8, mask);
__m256i wsa24_8 = _mm256_and_si256(ws24_8, mask);
__m256 l0_8 = _mm256_cvtepi32_ps(wsa0_8);
__m256 l8_8 = _mm256_cvtepi32_ps(wsa8_8);
__m256 l16_8 = _mm256_cvtepi32_ps(wsa16_8);
__m256 l24_8 = _mm256_cvtepi32_ps(wsa24_8);
acc0_0 = _mm256_fmadd_ps(v0_8, l0_8, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_8, l8_8, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_8, l16_8, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_8, l24_8, acc0_24);
__m256i ws0_9 = _mm256_srli_epi32(w0_0, 27);
__m256i ws8_9 = _mm256_srli_epi32(w0_8, 27);
__m256i ws16_9 = _mm256_srli_epi32(w0_16, 27);
__m256i ws24_9 = _mm256_srli_epi32(w0_24, 27);
__m256i wsa0_9 = _mm256_and_si256(ws0_9, mask);
__m256i wsa8_9 = _mm256_and_si256(ws8_9, mask);
__m256i wsa16_9 = _mm256_and_si256(ws16_9, mask);
__m256i wsa24_9 = _mm256_and_si256(ws24_9, mask);
__m256 l0_9 = _mm256_cvtepi32_ps(wsa0_9);
__m256 l8_9 = _mm256_cvtepi32_ps(wsa8_9);
__m256 l16_9 = _mm256_cvtepi32_ps(wsa16_9);
__m256 l24_9 = _mm256_cvtepi32_ps(wsa24_9);
acc0_0 = _mm256_fmadd_ps(v0_9, l0_9, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_9, l8_9, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_9, l16_9, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_9, l24_9, acc0_24);
__m256 v0_10 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+10)*nb + i1+0]);
__m256i ws0_10 = _mm256_srli_epi32(w0_0, 30);
__m256i temp0_0 = _mm256_slli_epi32(w1_0, 2);
temp0_0 = _mm256_and_si256(temp0_0, mask);
ws0_10 = _mm256_or_si256(ws0_10, temp0_0);
__m256i wsa0_10 = _mm256_and_si256(ws0_10, mask);
__m256 l0_10 = _mm256_cvtepi32_ps(wsa0_10);
acc0_0 = _mm256_fmadd_ps(v0_10, l0_10, acc0_0);
__m256i ws8_10 = _mm256_srli_epi32(w0_8, 30);
__m256i temp0_8 = _mm256_slli_epi32(w1_8, 2);
temp0_8 = _mm256_and_si256(temp0_8, mask);
ws8_10 = _mm256_or_si256(ws8_10, temp0_8);
__m256i wsa8_10 = _mm256_and_si256(ws8_10, mask);
__m256 l8_10 = _mm256_cvtepi32_ps(wsa8_10);
acc0_8 = _mm256_fmadd_ps(v0_10, l8_10, acc0_8);
__m256i ws16_10 = _mm256_srli_epi32(w0_16, 30);
__m256i temp0_16 = _mm256_slli_epi32(w1_16, 2);
temp0_16 = _mm256_and_si256(temp0_16, mask);
ws16_10 = _mm256_or_si256(ws16_10, temp0_16);
__m256i wsa16_10 = _mm256_and_si256(ws16_10, mask);
__m256 l16_10 = _mm256_cvtepi32_ps(wsa16_10);
acc0_16 = _mm256_fmadd_ps(v0_10, l16_10, acc0_16);
__m256i ws24_10 = _mm256_srli_epi32(w0_24, 30);
__m256i temp0_24 = _mm256_slli_epi32(w1_24, 2);
temp0_24 = _mm256_and_si256(temp0_24, mask);
ws24_10 = _mm256_or_si256(ws24_10, temp0_24);
__m256i wsa24_10 = _mm256_and_si256(ws24_10, mask);
__m256 l24_10 = _mm256_cvtepi32_ps(wsa24_10);
acc0_24 = _mm256_fmadd_ps(v0_10, l24_10, acc0_24);
__m256 v0_11 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+11)*nb + i1+0]);
__m256 v0_12 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+12)*nb + i1+0]);
__m256i ws0_11 = _mm256_srli_epi32(w1_0, 1);
__m256i ws8_11 = _mm256_srli_epi32(w1_8, 1);
__m256i ws16_11 = _mm256_srli_epi32(w1_16, 1);
__m256i ws24_11 = _mm256_srli_epi32(w1_24, 1);
__m256i wsa0_11 = _mm256_and_si256(ws0_11, mask);
__m256i wsa8_11 = _mm256_and_si256(ws8_11, mask);
__m256i wsa16_11 = _mm256_and_si256(ws16_11, mask);
__m256i wsa24_11 = _mm256_and_si256(ws24_11, mask);
__m256 l0_11 = _mm256_cvtepi32_ps(wsa0_11);
__m256 l8_11 = _mm256_cvtepi32_ps(wsa8_11);
__m256 l16_11 = _mm256_cvtepi32_ps(wsa16_11);
__m256 l24_11 = _mm256_cvtepi32_ps(wsa24_11);
acc0_0 = _mm256_fmadd_ps(v0_11, l0_11, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_11, l8_11, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_11, l16_11, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_11, l24_11, acc0_24);
__m256i ws0_12 = _mm256_srli_epi32(w1_0, 4);
__m256i ws8_12 = _mm256_srli_epi32(w1_8, 4);
__m256i ws16_12 = _mm256_srli_epi32(w1_16, 4);
__m256i ws24_12 = _mm256_srli_epi32(w1_24, 4);
__m256i wsa0_12 = _mm256_and_si256(ws0_12, mask);
__m256i wsa8_12 = _mm256_and_si256(ws8_12, mask);
__m256i wsa16_12 = _mm256_and_si256(ws16_12, mask);
__m256i wsa24_12 = _mm256_and_si256(ws24_12, mask);
__m256 l0_12 = _mm256_cvtepi32_ps(wsa0_12);
__m256 l8_12 = _mm256_cvtepi32_ps(wsa8_12);
__m256 l16_12 = _mm256_cvtepi32_ps(wsa16_12);
__m256 l24_12 = _mm256_cvtepi32_ps(wsa24_12);
acc0_0 = _mm256_fmadd_ps(v0_12, l0_12, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_12, l8_12, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_12, l16_12, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_12, l24_12, acc0_24);
__m256 v0_13 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+13)*nb + i1+0]);
__m256 v0_14 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+14)*nb + i1+0]);
__m256i ws0_13 = _mm256_srli_epi32(w1_0, 7);
__m256i ws8_13 = _mm256_srli_epi32(w1_8, 7);
__m256i ws16_13 = _mm256_srli_epi32(w1_16, 7);
__m256i ws24_13 = _mm256_srli_epi32(w1_24, 7);
__m256i wsa0_13 = _mm256_and_si256(ws0_13, mask);
__m256i wsa8_13 = _mm256_and_si256(ws8_13, mask);
__m256i wsa16_13 = _mm256_and_si256(ws16_13, mask);
__m256i wsa24_13 = _mm256_and_si256(ws24_13, mask);
__m256 l0_13 = _mm256_cvtepi32_ps(wsa0_13);
__m256 l8_13 = _mm256_cvtepi32_ps(wsa8_13);
__m256 l16_13 = _mm256_cvtepi32_ps(wsa16_13);
__m256 l24_13 = _mm256_cvtepi32_ps(wsa24_13);
acc0_0 = _mm256_fmadd_ps(v0_13, l0_13, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_13, l8_13, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_13, l16_13, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_13, l24_13, acc0_24);
__m256i ws0_14 = _mm256_srli_epi32(w1_0, 10);
__m256i ws8_14 = _mm256_srli_epi32(w1_8, 10);
__m256i ws16_14 = _mm256_srli_epi32(w1_16, 10);
__m256i ws24_14 = _mm256_srli_epi32(w1_24, 10);
__m256i wsa0_14 = _mm256_and_si256(ws0_14, mask);
__m256i wsa8_14 = _mm256_and_si256(ws8_14, mask);
__m256i wsa16_14 = _mm256_and_si256(ws16_14, mask);
__m256i wsa24_14 = _mm256_and_si256(ws24_14, mask);
__m256 l0_14 = _mm256_cvtepi32_ps(wsa0_14);
__m256 l8_14 = _mm256_cvtepi32_ps(wsa8_14);
__m256 l16_14 = _mm256_cvtepi32_ps(wsa16_14);
__m256 l24_14 = _mm256_cvtepi32_ps(wsa24_14);
acc0_0 = _mm256_fmadd_ps(v0_14, l0_14, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_14, l8_14, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_14, l16_14, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_14, l24_14, acc0_24);
__m256 v0_15 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+15)*nb + i1+0]);
__m256 v0_16 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+16)*nb + i1+0]);
__m256i ws0_15 = _mm256_srli_epi32(w1_0, 13);
__m256i ws8_15 = _mm256_srli_epi32(w1_8, 13);
__m256i ws16_15 = _mm256_srli_epi32(w1_16, 13);
__m256i ws24_15 = _mm256_srli_epi32(w1_24, 13);
__m256i wsa0_15 = _mm256_and_si256(ws0_15, mask);
__m256i wsa8_15 = _mm256_and_si256(ws8_15, mask);
__m256i wsa16_15 = _mm256_and_si256(ws16_15, mask);
__m256i wsa24_15 = _mm256_and_si256(ws24_15, mask);
__m256 l0_15 = _mm256_cvtepi32_ps(wsa0_15);
__m256 l8_15 = _mm256_cvtepi32_ps(wsa8_15);
__m256 l16_15 = _mm256_cvtepi32_ps(wsa16_15);
__m256 l24_15 = _mm256_cvtepi32_ps(wsa24_15);
acc0_0 = _mm256_fmadd_ps(v0_15, l0_15, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_15, l8_15, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_15, l16_15, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_15, l24_15, acc0_24);
__m256i ws0_16 = _mm256_srli_epi32(w1_0, 16);
__m256i ws8_16 = _mm256_srli_epi32(w1_8, 16);
__m256i ws16_16 = _mm256_srli_epi32(w1_16, 16);
__m256i ws24_16 = _mm256_srli_epi32(w1_24, 16);
__m256i wsa0_16 = _mm256_and_si256(ws0_16, mask);
__m256i wsa8_16 = _mm256_and_si256(ws8_16, mask);
__m256i wsa16_16 = _mm256_and_si256(ws16_16, mask);
__m256i wsa24_16 = _mm256_and_si256(ws24_16, mask);
__m256 l0_16 = _mm256_cvtepi32_ps(wsa0_16);
__m256 l8_16 = _mm256_cvtepi32_ps(wsa8_16);
__m256 l16_16 = _mm256_cvtepi32_ps(wsa16_16);
__m256 l24_16 = _mm256_cvtepi32_ps(wsa24_16);
acc0_0 = _mm256_fmadd_ps(v0_16, l0_16, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_16, l8_16, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_16, l16_16, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_16, l24_16, acc0_24);
__m256 v0_17 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+17)*nb + i1+0]);
__m256 v0_18 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+18)*nb + i1+0]);
__m256i ws0_17 = _mm256_srli_epi32(w1_0, 19);
__m256i ws8_17 = _mm256_srli_epi32(w1_8, 19);
__m256i ws16_17 = _mm256_srli_epi32(w1_16, 19);
__m256i ws24_17 = _mm256_srli_epi32(w1_24, 19);
__m256i wsa0_17 = _mm256_and_si256(ws0_17, mask);
__m256i wsa8_17 = _mm256_and_si256(ws8_17, mask);
__m256i wsa16_17 = _mm256_and_si256(ws16_17, mask);
__m256i wsa24_17 = _mm256_and_si256(ws24_17, mask);
__m256 l0_17 = _mm256_cvtepi32_ps(wsa0_17);
__m256 l8_17 = _mm256_cvtepi32_ps(wsa8_17);
__m256 l16_17 = _mm256_cvtepi32_ps(wsa16_17);
__m256 l24_17 = _mm256_cvtepi32_ps(wsa24_17);
acc0_0 = _mm256_fmadd_ps(v0_17, l0_17, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_17, l8_17, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_17, l16_17, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_17, l24_17, acc0_24);
__m256i ws0_18 = _mm256_srli_epi32(w1_0, 22);
__m256i ws8_18 = _mm256_srli_epi32(w1_8, 22);
__m256i ws16_18 = _mm256_srli_epi32(w1_16, 22);
__m256i ws24_18 = _mm256_srli_epi32(w1_24, 22);
__m256i wsa0_18 = _mm256_and_si256(ws0_18, mask);
__m256i wsa8_18 = _mm256_and_si256(ws8_18, mask);
__m256i wsa16_18 = _mm256_and_si256(ws16_18, mask);
__m256i wsa24_18 = _mm256_and_si256(ws24_18, mask);
__m256 l0_18 = _mm256_cvtepi32_ps(wsa0_18);
__m256 l8_18 = _mm256_cvtepi32_ps(wsa8_18);
__m256 l16_18 = _mm256_cvtepi32_ps(wsa16_18);
__m256 l24_18 = _mm256_cvtepi32_ps(wsa24_18);
acc0_0 = _mm256_fmadd_ps(v0_18, l0_18, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_18, l8_18, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_18, l16_18, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_18, l24_18, acc0_24);
__m256 v0_19 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+19)*nb + i1+0]);
__m256 v0_20 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+20)*nb + i1+0]);
__m256i ws0_19 = _mm256_srli_epi32(w1_0, 25);
__m256i ws8_19 = _mm256_srli_epi32(w1_8, 25);
__m256i ws16_19 = _mm256_srli_epi32(w1_16, 25);
__m256i ws24_19 = _mm256_srli_epi32(w1_24, 25);
__m256i wsa0_19 = _mm256_and_si256(ws0_19, mask);
__m256i wsa8_19 = _mm256_and_si256(ws8_19, mask);
__m256i wsa16_19 = _mm256_and_si256(ws16_19, mask);
__m256i wsa24_19 = _mm256_and_si256(ws24_19, mask);
__m256 l0_19 = _mm256_cvtepi32_ps(wsa0_19);
__m256 l8_19 = _mm256_cvtepi32_ps(wsa8_19);
__m256 l16_19 = _mm256_cvtepi32_ps(wsa16_19);
__m256 l24_19 = _mm256_cvtepi32_ps(wsa24_19);
acc0_0 = _mm256_fmadd_ps(v0_19, l0_19, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_19, l8_19, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_19, l16_19, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_19, l24_19, acc0_24);
__m256i ws0_20 = _mm256_srli_epi32(w1_0, 28);
__m256i ws8_20 = _mm256_srli_epi32(w1_8, 28);
__m256i ws16_20 = _mm256_srli_epi32(w1_16, 28);
__m256i ws24_20 = _mm256_srli_epi32(w1_24, 28);
__m256i wsa0_20 = _mm256_and_si256(ws0_20, mask);
__m256i wsa8_20 = _mm256_and_si256(ws8_20, mask);
__m256i wsa16_20 = _mm256_and_si256(ws16_20, mask);
__m256i wsa24_20 = _mm256_and_si256(ws24_20, mask);
__m256 l0_20 = _mm256_cvtepi32_ps(wsa0_20);
__m256 l8_20 = _mm256_cvtepi32_ps(wsa8_20);
__m256 l16_20 = _mm256_cvtepi32_ps(wsa16_20);
__m256 l24_20 = _mm256_cvtepi32_ps(wsa24_20);
acc0_0 = _mm256_fmadd_ps(v0_20, l0_20, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_20, l8_20, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_20, l16_20, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_20, l24_20, acc0_24);
__m256 v0_21 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+21)*nb + i1+0]);
__m256i ws0_21 = _mm256_srli_epi32(w1_0, 31);
__m256i temp1_0 = _mm256_slli_epi32(w2_0, 1);
temp1_0 = _mm256_and_si256(temp1_0, mask);
ws0_21 = _mm256_or_si256(ws0_21, temp1_0);
__m256i wsa0_21 = _mm256_and_si256(ws0_21, mask);
__m256 l0_21 = _mm256_cvtepi32_ps(wsa0_21);
acc0_0 = _mm256_fmadd_ps(v0_21, l0_21, acc0_0);
__m256i ws8_21 = _mm256_srli_epi32(w1_8, 31);
__m256i temp1_8 = _mm256_slli_epi32(w2_8, 1);
temp1_8 = _mm256_and_si256(temp1_8, mask);
ws8_21 = _mm256_or_si256(ws8_21, temp1_8);
__m256i wsa8_21 = _mm256_and_si256(ws8_21, mask);
__m256 l8_21 = _mm256_cvtepi32_ps(wsa8_21);
acc0_8 = _mm256_fmadd_ps(v0_21, l8_21, acc0_8);
__m256i ws16_21 = _mm256_srli_epi32(w1_16, 31);
__m256i temp1_16 = _mm256_slli_epi32(w2_16, 1);
temp1_16 = _mm256_and_si256(temp1_16, mask);
ws16_21 = _mm256_or_si256(ws16_21, temp1_16);
__m256i wsa16_21 = _mm256_and_si256(ws16_21, mask);
__m256 l16_21 = _mm256_cvtepi32_ps(wsa16_21);
acc0_16 = _mm256_fmadd_ps(v0_21, l16_21, acc0_16);
__m256i ws24_21 = _mm256_srli_epi32(w1_24, 31);
__m256i temp1_24 = _mm256_slli_epi32(w2_24, 1);
temp1_24 = _mm256_and_si256(temp1_24, mask);
ws24_21 = _mm256_or_si256(ws24_21, temp1_24);
__m256i wsa24_21 = _mm256_and_si256(ws24_21, mask);
__m256 l24_21 = _mm256_cvtepi32_ps(wsa24_21);
acc0_24 = _mm256_fmadd_ps(v0_21, l24_21, acc0_24);
__m256 v0_22 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+22)*nb + i1+0]);
__m256 v0_23 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+23)*nb + i1+0]);
__m256i ws0_22 = _mm256_srli_epi32(w2_0, 2);
__m256i ws8_22 = _mm256_srli_epi32(w2_8, 2);
__m256i ws16_22 = _mm256_srli_epi32(w2_16, 2);
__m256i ws24_22 = _mm256_srli_epi32(w2_24, 2);
__m256i wsa0_22 = _mm256_and_si256(ws0_22, mask);
__m256i wsa8_22 = _mm256_and_si256(ws8_22, mask);
__m256i wsa16_22 = _mm256_and_si256(ws16_22, mask);
__m256i wsa24_22 = _mm256_and_si256(ws24_22, mask);
__m256 l0_22 = _mm256_cvtepi32_ps(wsa0_22);
__m256 l8_22 = _mm256_cvtepi32_ps(wsa8_22);
__m256 l16_22 = _mm256_cvtepi32_ps(wsa16_22);
__m256 l24_22 = _mm256_cvtepi32_ps(wsa24_22);
acc0_0 = _mm256_fmadd_ps(v0_22, l0_22, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_22, l8_22, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_22, l16_22, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_22, l24_22, acc0_24);
__m256i ws0_23 = _mm256_srli_epi32(w2_0, 5);
__m256i ws8_23 = _mm256_srli_epi32(w2_8, 5);
__m256i ws16_23 = _mm256_srli_epi32(w2_16, 5);
__m256i ws24_23 = _mm256_srli_epi32(w2_24, 5);
__m256i wsa0_23 = _mm256_and_si256(ws0_23, mask);
__m256i wsa8_23 = _mm256_and_si256(ws8_23, mask);
__m256i wsa16_23 = _mm256_and_si256(ws16_23, mask);
__m256i wsa24_23 = _mm256_and_si256(ws24_23, mask);
__m256 l0_23 = _mm256_cvtepi32_ps(wsa0_23);
__m256 l8_23 = _mm256_cvtepi32_ps(wsa8_23);
__m256 l16_23 = _mm256_cvtepi32_ps(wsa16_23);
__m256 l24_23 = _mm256_cvtepi32_ps(wsa24_23);
acc0_0 = _mm256_fmadd_ps(v0_23, l0_23, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_23, l8_23, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_23, l16_23, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_23, l24_23, acc0_24);
__m256 v0_24 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+24)*nb + i1+0]);
__m256 v0_25 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+25)*nb + i1+0]);
__m256i ws0_24 = _mm256_srli_epi32(w2_0, 8);
__m256i ws8_24 = _mm256_srli_epi32(w2_8, 8);
__m256i ws16_24 = _mm256_srli_epi32(w2_16, 8);
__m256i ws24_24 = _mm256_srli_epi32(w2_24, 8);
__m256i wsa0_24 = _mm256_and_si256(ws0_24, mask);
__m256i wsa8_24 = _mm256_and_si256(ws8_24, mask);
__m256i wsa16_24 = _mm256_and_si256(ws16_24, mask);
__m256i wsa24_24 = _mm256_and_si256(ws24_24, mask);
__m256 l0_24 = _mm256_cvtepi32_ps(wsa0_24);
__m256 l8_24 = _mm256_cvtepi32_ps(wsa8_24);
__m256 l16_24 = _mm256_cvtepi32_ps(wsa16_24);
__m256 l24_24 = _mm256_cvtepi32_ps(wsa24_24);
acc0_0 = _mm256_fmadd_ps(v0_24, l0_24, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_24, l8_24, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_24, l16_24, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_24, l24_24, acc0_24);
__m256i ws0_25 = _mm256_srli_epi32(w2_0, 11);
__m256i ws8_25 = _mm256_srli_epi32(w2_8, 11);
__m256i ws16_25 = _mm256_srli_epi32(w2_16, 11);
__m256i ws24_25 = _mm256_srli_epi32(w2_24, 11);
__m256i wsa0_25 = _mm256_and_si256(ws0_25, mask);
__m256i wsa8_25 = _mm256_and_si256(ws8_25, mask);
__m256i wsa16_25 = _mm256_and_si256(ws16_25, mask);
__m256i wsa24_25 = _mm256_and_si256(ws24_25, mask);
__m256 l0_25 = _mm256_cvtepi32_ps(wsa0_25);
__m256 l8_25 = _mm256_cvtepi32_ps(wsa8_25);
__m256 l16_25 = _mm256_cvtepi32_ps(wsa16_25);
__m256 l24_25 = _mm256_cvtepi32_ps(wsa24_25);
acc0_0 = _mm256_fmadd_ps(v0_25, l0_25, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_25, l8_25, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_25, l16_25, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_25, l24_25, acc0_24);
__m256 v0_26 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+26)*nb + i1+0]);
__m256 v0_27 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+27)*nb + i1+0]);
__m256i ws0_26 = _mm256_srli_epi32(w2_0, 14);
__m256i ws8_26 = _mm256_srli_epi32(w2_8, 14);
__m256i ws16_26 = _mm256_srli_epi32(w2_16, 14);
__m256i ws24_26 = _mm256_srli_epi32(w2_24, 14);
__m256i wsa0_26 = _mm256_and_si256(ws0_26, mask);
__m256i wsa8_26 = _mm256_and_si256(ws8_26, mask);
__m256i wsa16_26 = _mm256_and_si256(ws16_26, mask);
__m256i wsa24_26 = _mm256_and_si256(ws24_26, mask);
__m256 l0_26 = _mm256_cvtepi32_ps(wsa0_26);
__m256 l8_26 = _mm256_cvtepi32_ps(wsa8_26);
__m256 l16_26 = _mm256_cvtepi32_ps(wsa16_26);
__m256 l24_26 = _mm256_cvtepi32_ps(wsa24_26);
acc0_0 = _mm256_fmadd_ps(v0_26, l0_26, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_26, l8_26, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_26, l16_26, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_26, l24_26, acc0_24);
__m256i ws0_27 = _mm256_srli_epi32(w2_0, 17);
__m256i ws8_27 = _mm256_srli_epi32(w2_8, 17);
__m256i ws16_27 = _mm256_srli_epi32(w2_16, 17);
__m256i ws24_27 = _mm256_srli_epi32(w2_24, 17);
__m256i wsa0_27 = _mm256_and_si256(ws0_27, mask);
__m256i wsa8_27 = _mm256_and_si256(ws8_27, mask);
__m256i wsa16_27 = _mm256_and_si256(ws16_27, mask);
__m256i wsa24_27 = _mm256_and_si256(ws24_27, mask);
__m256 l0_27 = _mm256_cvtepi32_ps(wsa0_27);
__m256 l8_27 = _mm256_cvtepi32_ps(wsa8_27);
__m256 l16_27 = _mm256_cvtepi32_ps(wsa16_27);
__m256 l24_27 = _mm256_cvtepi32_ps(wsa24_27);
acc0_0 = _mm256_fmadd_ps(v0_27, l0_27, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_27, l8_27, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_27, l16_27, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_27, l24_27, acc0_24);
__m256 v0_28 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+28)*nb + i1+0]);
__m256 v0_29 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+29)*nb + i1+0]);
__m256i ws0_28 = _mm256_srli_epi32(w2_0, 20);
__m256i ws8_28 = _mm256_srli_epi32(w2_8, 20);
__m256i ws16_28 = _mm256_srli_epi32(w2_16, 20);
__m256i ws24_28 = _mm256_srli_epi32(w2_24, 20);
__m256i wsa0_28 = _mm256_and_si256(ws0_28, mask);
__m256i wsa8_28 = _mm256_and_si256(ws8_28, mask);
__m256i wsa16_28 = _mm256_and_si256(ws16_28, mask);
__m256i wsa24_28 = _mm256_and_si256(ws24_28, mask);
__m256 l0_28 = _mm256_cvtepi32_ps(wsa0_28);
__m256 l8_28 = _mm256_cvtepi32_ps(wsa8_28);
__m256 l16_28 = _mm256_cvtepi32_ps(wsa16_28);
__m256 l24_28 = _mm256_cvtepi32_ps(wsa24_28);
acc0_0 = _mm256_fmadd_ps(v0_28, l0_28, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_28, l8_28, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_28, l16_28, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_28, l24_28, acc0_24);
__m256i ws0_29 = _mm256_srli_epi32(w2_0, 23);
__m256i ws8_29 = _mm256_srli_epi32(w2_8, 23);
__m256i ws16_29 = _mm256_srli_epi32(w2_16, 23);
__m256i ws24_29 = _mm256_srli_epi32(w2_24, 23);
__m256i wsa0_29 = _mm256_and_si256(ws0_29, mask);
__m256i wsa8_29 = _mm256_and_si256(ws8_29, mask);
__m256i wsa16_29 = _mm256_and_si256(ws16_29, mask);
__m256i wsa24_29 = _mm256_and_si256(ws24_29, mask);
__m256 l0_29 = _mm256_cvtepi32_ps(wsa0_29);
__m256 l8_29 = _mm256_cvtepi32_ps(wsa8_29);
__m256 l16_29 = _mm256_cvtepi32_ps(wsa16_29);
__m256 l24_29 = _mm256_cvtepi32_ps(wsa24_29);
acc0_0 = _mm256_fmadd_ps(v0_29, l0_29, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_29, l8_29, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_29, l16_29, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_29, l24_29, acc0_24);
__m256 v0_30 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+30)*nb + i1+0]);
__m256 v0_31 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+31)*nb + i1+0]);
__m256i ws0_30 = _mm256_srli_epi32(w2_0, 26);
__m256i ws8_30 = _mm256_srli_epi32(w2_8, 26);
__m256i ws16_30 = _mm256_srli_epi32(w2_16, 26);
__m256i ws24_30 = _mm256_srli_epi32(w2_24, 26);
__m256i wsa0_30 = _mm256_and_si256(ws0_30, mask);
__m256i wsa8_30 = _mm256_and_si256(ws8_30, mask);
__m256i wsa16_30 = _mm256_and_si256(ws16_30, mask);
__m256i wsa24_30 = _mm256_and_si256(ws24_30, mask);
__m256 l0_30 = _mm256_cvtepi32_ps(wsa0_30);
__m256 l8_30 = _mm256_cvtepi32_ps(wsa8_30);
__m256 l16_30 = _mm256_cvtepi32_ps(wsa16_30);
__m256 l24_30 = _mm256_cvtepi32_ps(wsa24_30);
acc0_0 = _mm256_fmadd_ps(v0_30, l0_30, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_30, l8_30, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_30, l16_30, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_30, l24_30, acc0_24);
__m256i ws0_31 = _mm256_srli_epi32(w2_0, 29);
__m256i ws8_31 = _mm256_srli_epi32(w2_8, 29);
__m256i ws16_31 = _mm256_srli_epi32(w2_16, 29);
__m256i ws24_31 = _mm256_srli_epi32(w2_24, 29);
__m256i wsa0_31 = _mm256_and_si256(ws0_31, mask);
__m256i wsa8_31 = _mm256_and_si256(ws8_31, mask);
__m256i wsa16_31 = _mm256_and_si256(ws16_31, mask);
__m256i wsa24_31 = _mm256_and_si256(ws24_31, mask);
__m256 l0_31 = _mm256_cvtepi32_ps(wsa0_31);
__m256 l8_31 = _mm256_cvtepi32_ps(wsa8_31);
__m256 l16_31 = _mm256_cvtepi32_ps(wsa16_31);
__m256 l24_31 = _mm256_cvtepi32_ps(wsa24_31);
acc0_0 = _mm256_fmadd_ps(v0_31, l0_31, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_31, l8_31, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_31, l16_31, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_31, l24_31, acc0_24);
}
__m256 o0_0 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+0]);
__m256 o0_8 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+8]);
__m256 o0_16 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+16]);
__m256 o0_24 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+24]);
__m256 s0_0 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+0]);
__m256 s0_8 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+8]);
__m256 s0_16 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+16]);
__m256 s0_24 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+24]);
__m256 f0_0 = _mm256_fmadd_ps(acc0_0, s0_0, o0_0);
__m256 f0_8 = _mm256_fmadd_ps(acc0_8, s0_8, o0_8);
__m256 f0_16 = _mm256_fmadd_ps(acc0_16, s0_16, o0_16);
__m256 f0_24 = _mm256_fmadd_ps(acc0_24, s0_24, o0_24);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+0], f0_0);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+8], f0_8);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+16], f0_16);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+24], f0_24);
}
}
}
}
}
}
#pragma omp barrier
const int ngs = m/gs;
for (int i = 0; i < n; i++) {
for (int j = 0; j < tt; j+=32){
__m256 acc0 = _mm256_setzero_ps();
__m256 acc8 = _mm256_setzero_ps();
__m256 acc16 = _mm256_setzero_ps();
__m256 acc24 = _mm256_setzero_ps();
for (int i1 = 0; i1 < ngs; i1++){
__m256 r = _mm256_set1_ps(sums[i*ngs + i1]);
__m256 z0 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 0]);
__m256 z8 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 8]);
__m256 z16 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 16]);
__m256 z24 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 24]);
acc0 = _mm256_fmadd_ps(z0, r, acc0);
acc8 = _mm256_fmadd_ps(z8, r, acc8);
acc16 = _mm256_fmadd_ps(z16, r, acc16);
acc24 = _mm256_fmadd_ps(z24, r, acc24);
}
__m256 o0 = _mm256_loadu_ps(&output[i*t + base_output + j + 0]);
__m256 o8 = _mm256_loadu_ps(&output[i*t + base_output + j + 8]);
__m256 o16 = _mm256_loadu_ps(&output[i*t + base_output + j + 16]);
__m256 o24 = _mm256_loadu_ps(&output[i*t + base_output + j + 24]);
__m256 b0 = _mm256_loadu_ps(&bias[base_output + j + 0]);
__m256 b8 = _mm256_loadu_ps(&bias[base_output + j + 8]);
__m256 b16 = _mm256_loadu_ps(&bias[base_output + j + 16]);
__m256 b24 = _mm256_loadu_ps(&bias[base_output + j + 24]);
__m256 o10 = _mm256_sub_ps(o0, acc0);
__m256 o18 = _mm256_sub_ps(o8, acc8);
__m256 o116 = _mm256_sub_ps(o16, acc16);
__m256 o124 = _mm256_sub_ps(o24, acc24);
__m256 o20 = _mm256_add_ps(o10, b0);
__m256 o28 = _mm256_add_ps(o18, b8);
__m256 o216 = _mm256_add_ps(o116, b16);
__m256 o224 = _mm256_add_ps(o124, b24);
_mm256_storeu_ps(&output[i*t + base_output + j + 0], o20);
_mm256_storeu_ps(&output[i*t + base_output + j + 8], o28);
_mm256_storeu_ps(&output[i*t + base_output + j + 16], o216);
_mm256_storeu_ps(&output[i*t + base_output + j + 24], o224);
}
}
}
}
inline void forward3_gs_cpu(
torch::Tensor in, torch::Tensor weight, torch::Tensor out,
torch::Tensor bias, torch::Tensor scales, torch::Tensor zeros, torch::Tensor sums,
int N, int M, int T, int nb, int mb, int tb, int tt, int groupsize, int cutoff){
int*   W = weight.data_ptr<int>();
float* input = in.data_ptr<float>();
float* b   = bias.data_ptr<float>();
float* s   = scales.data_ptr<float>();
float* z   = zeros.data_ptr<float>();
float* r   = sums.data_ptr<float>();
float* O   = out.data_ptr<float>();

q3gemm_gs(input, W, s, z, b, r, O, N, M, T, nb, mb, tb, tt, groupsize, cutoff);
}
inline void pack3_qw_inner(int* A, int* B, const int N, const int M, const int nb, const int mb, int cutoff){
// copy the full matrix A in blocked format into B
uint64_t idx = 0;
for(int j = 0, tid = 0; j < M; j+=mb, tid++){
for(int i = 0; i < N; i+=nb){
 for(int ii = i; ii < mymin(i+nb, N); ii+=3){
 for(int jj = j; jj < mymin(j+mb, M); jj+=8){
 for(int iii = ii; iii < ii + 3; iii++){
 for(int jjj = jj; jjj < jj + 8; jjj++){
 B[idx] = A[iii*M+jjj];
 idx++;
 }
 }
 }
 }
 }
 }
 }
inline void pack3_w_cpu(
torch::Tensor in, torch::Tensor out,
int N, int M, int nb, int mb, int cutoff){
int* input = in.data_ptr<int>();
int* O = out.data_ptr<int>();
pack3_qw_inner(input, O, N, M, nb, mb, cutoff);
}
void unpack_zeros3_cpu(const int* zv, float* ov, int n, int m){
const __m256i ones = _mm256_set1_epi32(1);
const __m256i mask = _mm256_set1_epi32(7);
for(int i = 0; i < n; i++){
for(int j = 0; j < m; j+=32){
std::cout<<"not yet implemented"<<std::endl;
}
}
}
void unpack_zeros3(torch::Tensor zeros, torch::Tensor out, int N, int M){
int* Z = zeros.data_ptr<int>();
float* O = out.data_ptr<float>();
unpack_zeros3_cpu(Z, O, N, M);
}
inline
void q4gemm(const float* __restrict__ input, 
const int* __restrict__ W, 
const float* __restrict__ scales, 
const float* __restrict__ zeros, 
const float* __restrict__ bias, 
 const float* __restrict__ sums, 
 float* __restrict__ output,
const int n,
const int m,
const int t,
const int nb,
const int mb,
const int tb,
int ogtt,
const int cutoff){
#pragma omp parallel num_threads(12)
{
int tid;
const int mu = 16;
const int nu = 1;
const int tu = 32;
const int on = n / nb;
const int om = m / mb;
const __m256i mask = _mm256_set1_epi32(15);
tid = omp_get_thread_num();
int tt = ogtt;
if(tid >= cutoff){
tt -= tb;
}
const int base_output = tid >= cutoff ?
 (tid-cutoff)*tt + (tt+tb)*cutoff: 
 tid*tt;
const int base_W = tid >= cutoff ?
 ((tid-cutoff)*tt + (tt+tb)*cutoff)*m/8: 
 tid*tt*m/8;
for(int j = 0; j < tt; j+=tb){
for(int i = 0; i < on; i++) {
for(int k = 0; k < om; k++) {
for(int i1 = 0; i1 < nb; i1+=nu) {
int j1 = 0;
for(; j1 < tb-tu+1; j1+=tu) {
__m256 acc0_0 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+0]);
__m256 acc0_8 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+8]);
__m256 acc0_16 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+16]);
__m256 acc0_24 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+24]);
for(int k1 = 0; k1 < mb; k1+=mu) {
for(int k2 = k1; k2 < k1+mu; k2+=8){
__m256i w0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/8 + k*mb*tb/8 + k2*tb/8 + j1+0]);
__m256i w8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/8 + k*mb*tb/8 + k2*tb/8 + j1+8]);
__m256i w16 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/8 + k*mb*tb/8 + k2*tb/8 + j1+16]);
__m256i w24 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/8 + k*mb*tb/8 + k2*tb/8 + j1+24]);
__m256 v0_7 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+7)*nb + i1+0]);
__m256 v0_6 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+6)*nb + i1+0]);
__m256i ws0_6 = _mm256_srli_epi32(w0, 24);
__m256i ws8_6 = _mm256_srli_epi32(w8, 24);
__m256i ws16_6 = _mm256_srli_epi32(w16, 24);
__m256i ws24_6 = _mm256_srli_epi32(w24, 24);
__m256i wsa0_6= _mm256_and_si256(ws0_6, mask);
__m256i wsa8_6= _mm256_and_si256(ws8_6, mask);
__m256i wsa16_6= _mm256_and_si256(ws16_6, mask);
__m256i wsa24_6= _mm256_and_si256(ws24_6, mask);
__m256 l0_6 = _mm256_cvtepi32_ps(wsa0_6);
__m256 l8_6 = _mm256_cvtepi32_ps(wsa8_6);
__m256 l16_6 = _mm256_cvtepi32_ps(wsa16_6);
__m256 l24_6 = _mm256_cvtepi32_ps(wsa24_6);
acc0_0 = _mm256_fmadd_ps(v0_6, l0_6, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_6, l8_6, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_6, l16_6, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_6, l24_6, acc0_24);
__m256i ws0_7 = _mm256_srli_epi32(w0, 28);
__m256i ws8_7 = _mm256_srli_epi32(w8, 28);
__m256i ws16_7 = _mm256_srli_epi32(w16, 28);
__m256i ws24_7 = _mm256_srli_epi32(w24, 28);
__m256i wsa0_7= _mm256_and_si256(ws0_7, mask);
__m256i wsa8_7= _mm256_and_si256(ws8_7, mask);
__m256i wsa16_7= _mm256_and_si256(ws16_7, mask);
__m256i wsa24_7= _mm256_and_si256(ws24_7, mask);
__m256 l0_7 = _mm256_cvtepi32_ps(wsa0_7);
__m256 l8_7 = _mm256_cvtepi32_ps(wsa8_7);
__m256 l16_7 = _mm256_cvtepi32_ps(wsa16_7);
__m256 l24_7 = _mm256_cvtepi32_ps(wsa24_7);
acc0_0 = _mm256_fmadd_ps(v0_7, l0_7, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_7, l8_7, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_7, l16_7, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_7, l24_7, acc0_24);
__m256 v0_5 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+5)*nb + i1+0]);
__m256 v0_4 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+4)*nb + i1+0]);
__m256i ws0_4 = _mm256_srli_epi32(w0, 16);
__m256i ws8_4 = _mm256_srli_epi32(w8, 16);
__m256i ws16_4 = _mm256_srli_epi32(w16, 16);
__m256i ws24_4 = _mm256_srli_epi32(w24, 16);
__m256i wsa0_4= _mm256_and_si256(ws0_4, mask);
__m256i wsa8_4= _mm256_and_si256(ws8_4, mask);
__m256i wsa16_4= _mm256_and_si256(ws16_4, mask);
__m256i wsa24_4= _mm256_and_si256(ws24_4, mask);
__m256 l0_4 = _mm256_cvtepi32_ps(wsa0_4);
__m256 l8_4 = _mm256_cvtepi32_ps(wsa8_4);
__m256 l16_4 = _mm256_cvtepi32_ps(wsa16_4);
__m256 l24_4 = _mm256_cvtepi32_ps(wsa24_4);
acc0_0 = _mm256_fmadd_ps(v0_4, l0_4, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_4, l8_4, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_4, l16_4, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_4, l24_4, acc0_24);
__m256i ws0_5 = _mm256_srli_epi32(w0, 20);
__m256i ws8_5 = _mm256_srli_epi32(w8, 20);
__m256i ws16_5 = _mm256_srli_epi32(w16, 20);
__m256i ws24_5 = _mm256_srli_epi32(w24, 20);
__m256i wsa0_5= _mm256_and_si256(ws0_5, mask);
__m256i wsa8_5= _mm256_and_si256(ws8_5, mask);
__m256i wsa16_5= _mm256_and_si256(ws16_5, mask);
__m256i wsa24_5= _mm256_and_si256(ws24_5, mask);
__m256 l0_5 = _mm256_cvtepi32_ps(wsa0_5);
__m256 l8_5 = _mm256_cvtepi32_ps(wsa8_5);
__m256 l16_5 = _mm256_cvtepi32_ps(wsa16_5);
__m256 l24_5 = _mm256_cvtepi32_ps(wsa24_5);
acc0_0 = _mm256_fmadd_ps(v0_5, l0_5, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_5, l8_5, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_5, l16_5, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_5, l24_5, acc0_24);
__m256 v0_3 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+3)*nb + i1+0]);
__m256 v0_2 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+2)*nb + i1+0]);
__m256i ws0_2 = _mm256_srli_epi32(w0, 8);
__m256i ws8_2 = _mm256_srli_epi32(w8, 8);
__m256i ws16_2 = _mm256_srli_epi32(w16, 8);
__m256i ws24_2 = _mm256_srli_epi32(w24, 8);
__m256i wsa0_2= _mm256_and_si256(ws0_2, mask);
__m256i wsa8_2= _mm256_and_si256(ws8_2, mask);
__m256i wsa16_2= _mm256_and_si256(ws16_2, mask);
__m256i wsa24_2= _mm256_and_si256(ws24_2, mask);
__m256 l0_2 = _mm256_cvtepi32_ps(wsa0_2);
__m256 l8_2 = _mm256_cvtepi32_ps(wsa8_2);
__m256 l16_2 = _mm256_cvtepi32_ps(wsa16_2);
__m256 l24_2 = _mm256_cvtepi32_ps(wsa24_2);
acc0_0 = _mm256_fmadd_ps(v0_2, l0_2, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_2, l8_2, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_2, l16_2, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_2, l24_2, acc0_24);
__m256i ws0_3 = _mm256_srli_epi32(w0, 12);
__m256i ws8_3 = _mm256_srli_epi32(w8, 12);
__m256i ws16_3 = _mm256_srli_epi32(w16, 12);
__m256i ws24_3 = _mm256_srli_epi32(w24, 12);
__m256i wsa0_3= _mm256_and_si256(ws0_3, mask);
__m256i wsa8_3= _mm256_and_si256(ws8_3, mask);
__m256i wsa16_3= _mm256_and_si256(ws16_3, mask);
__m256i wsa24_3= _mm256_and_si256(ws24_3, mask);
__m256 l0_3 = _mm256_cvtepi32_ps(wsa0_3);
__m256 l8_3 = _mm256_cvtepi32_ps(wsa8_3);
__m256 l16_3 = _mm256_cvtepi32_ps(wsa16_3);
__m256 l24_3 = _mm256_cvtepi32_ps(wsa24_3);
acc0_0 = _mm256_fmadd_ps(v0_3, l0_3, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_3, l8_3, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_3, l16_3, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_3, l24_3, acc0_24);
__m256 v0_1 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+1)*nb + i1+0]);
__m256 v0_0 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+0)*nb + i1+0]);
__m256i ws0_0 = _mm256_srli_epi32(w0, 0);
__m256i ws8_0 = _mm256_srli_epi32(w8, 0);
__m256i ws16_0 = _mm256_srli_epi32(w16, 0);
__m256i ws24_0 = _mm256_srli_epi32(w24, 0);
__m256i wsa0_0= _mm256_and_si256(ws0_0, mask);
__m256i wsa8_0= _mm256_and_si256(ws8_0, mask);
__m256i wsa16_0= _mm256_and_si256(ws16_0, mask);
__m256i wsa24_0= _mm256_and_si256(ws24_0, mask);
__m256 l0_0 = _mm256_cvtepi32_ps(wsa0_0);
__m256 l8_0 = _mm256_cvtepi32_ps(wsa8_0);
__m256 l16_0 = _mm256_cvtepi32_ps(wsa16_0);
__m256 l24_0 = _mm256_cvtepi32_ps(wsa24_0);
acc0_0 = _mm256_fmadd_ps(v0_0, l0_0, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_0, l8_0, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_0, l16_0, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_0, l24_0, acc0_24);
__m256i ws0_1 = _mm256_srli_epi32(w0, 4);
__m256i ws8_1 = _mm256_srli_epi32(w8, 4);
__m256i ws16_1 = _mm256_srli_epi32(w16, 4);
__m256i ws24_1 = _mm256_srli_epi32(w24, 4);
__m256i wsa0_1= _mm256_and_si256(ws0_1, mask);
__m256i wsa8_1= _mm256_and_si256(ws8_1, mask);
__m256i wsa16_1= _mm256_and_si256(ws16_1, mask);
__m256i wsa24_1= _mm256_and_si256(ws24_1, mask);
__m256 l0_1 = _mm256_cvtepi32_ps(wsa0_1);
__m256 l8_1 = _mm256_cvtepi32_ps(wsa8_1);
__m256 l16_1 = _mm256_cvtepi32_ps(wsa16_1);
__m256 l24_1 = _mm256_cvtepi32_ps(wsa24_1);
acc0_0 = _mm256_fmadd_ps(v0_1, l0_1, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_1, l8_1, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_1, l16_1, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_1, l24_1, acc0_24);
}
}
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+0], acc0_0);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+8], acc0_8);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+16], acc0_16);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+24], acc0_24);
}
}
}
}
}
#pragma omp barrier
for (int i = 0; i < n; i++) {
__m256 r = _mm256_set1_ps(sums[i]);
for (int j = 0; j < tt; j+=32){
__m256 o0 = _mm256_loadu_ps(&output[i*t + base_output + j + 0]);
__m256 o8 = _mm256_loadu_ps(&output[i*t + base_output + j + 8]);
__m256 o16 = _mm256_loadu_ps(&output[i*t + base_output + j + 16]);
__m256 o24 = _mm256_loadu_ps(&output[i*t + base_output + j + 24]);
__m256 z0 = _mm256_loadu_ps(&zeros[base_output + j + 0]);
__m256 z8 = _mm256_loadu_ps(&zeros[base_output + j + 8]);
__m256 z16 = _mm256_loadu_ps(&zeros[base_output + j + 16]);
__m256 z24 = _mm256_loadu_ps(&zeros[base_output + j + 24]);
__m256 b0 = _mm256_loadu_ps(&bias[base_output + j + 0]);
__m256 b8 = _mm256_loadu_ps(&bias[base_output + j + 8]);
__m256 b16 = _mm256_loadu_ps(&bias[base_output + j + 16]);
__m256 b24 = _mm256_loadu_ps(&bias[base_output + j + 24]);
__m256 s0 = _mm256_loadu_ps(&scales[base_output + j + 0]);
__m256 s8 = _mm256_loadu_ps(&scales[base_output + j + 8]);
__m256 s16 = _mm256_loadu_ps(&scales[base_output + j + 16]);
__m256 s24 = _mm256_loadu_ps(&scales[base_output + j + 24]);
__m256 zr0 = _mm256_fnmadd_ps(z0, r, o0);
__m256 zr8 = _mm256_fnmadd_ps(z8, r, o8);
__m256 zr16 = _mm256_fnmadd_ps(z16, r, o16);
__m256 zr24 = _mm256_fnmadd_ps(z24, r, o24);
__m256 o20 = _mm256_fmadd_ps(zr0, s0, b0);
__m256 o28 = _mm256_fmadd_ps(zr8, s8, b8);
__m256 o216 = _mm256_fmadd_ps(zr16, s16, b16);
__m256 o224 = _mm256_fmadd_ps(zr24, s24, b24);
_mm256_storeu_ps(&output[i*t + base_output + j + 0], o20);
_mm256_storeu_ps(&output[i*t + base_output + j + 8], o28);
_mm256_storeu_ps(&output[i*t + base_output + j + 16], o216);
_mm256_storeu_ps(&output[i*t + base_output + j + 24], o224);
}
}
}
}
inline void forward4_cpu(
torch::Tensor in, torch::Tensor weight, torch::Tensor out,
torch::Tensor bias, torch::Tensor scales, torch::Tensor zeros, torch::Tensor sums,
int N, int M, int T, int nb, int mb, int tb, int tt, int cutoff){
int*   W = weight.data_ptr<int>();
float* input = in.data_ptr<float>();
float* b   = bias.data_ptr<float>();
float* s   = scales.data_ptr<float>();
float* z   = zeros.data_ptr<float>();
float* r   = sums.data_ptr<float>();
float* O   = out.data_ptr<float>();

q4gemm(input, W, s, z, b, r, O, N, M, T, nb, mb, tb, tt, cutoff);
}
inline
void q4gemm_gs(const float* __restrict__ input, 
const int* __restrict__ W, 
const float* __restrict__ scales, 
const float* __restrict__ zeros, 
const float* __restrict__ bias, 
 const float* __restrict__ sums, 
 float* __restrict__ output,
const int n,
const int m,
const int t,
const int nb,
const int mb,
const int tb,
int ogtt,
const int gs,
const int cutoff){
#pragma omp parallel num_threads(12)
{
int tid;
const int mu = 16;
const int nu = 1;
const int tu = 32;
const int on = n / nb;
const int om = m / mb;
const __m256i mask = _mm256_set1_epi32(15);
tid = omp_get_thread_num();
int tt = ogtt;
if(tid >= cutoff){
tt -= tb;
}
const int base_output = tid >= cutoff ?
 (tid-cutoff)*tt + (tt+tb)*cutoff: 
 tid*tt;
const int base_W = tid >= cutoff ?
 ((tid-cutoff)*tt + (tt+tb)*cutoff)*m/8: 
 tid*tt*m/8;
for(int j = 0; j < tt; j+=tb){
for(int i = 0; i < on; i++) {
for(int k = 0; k < om; k++) {
for(int i1 = 0; i1 < nb; i1+=nu) {
int j1 = 0;
for(; j1 < tb-tu+1; j1+=tu) {
for(int k1 = 0; k1 < mb; k1+=gs) {
__m256 acc0_0 = _mm256_setzero_ps();
__m256 acc0_8 = _mm256_setzero_ps();
__m256 acc0_16 = _mm256_setzero_ps();
__m256 acc0_24 = _mm256_setzero_ps();
for(int k2 = k1; k2 < k1+gs; k2+=8)
{
__m256i w0 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/8 + k*mb*tb/8 + k2*tb/8 + j1+0]);
__m256i w8 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/8 + k*mb*tb/8 + k2*tb/8 + j1+8]);
__m256i w16 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/8 + k*mb*tb/8 + k2*tb/8 + j1+16]);
__m256i w24 = _mm256_loadu_si256((__m256i*)&W[base_W + j*m/8 + k*mb*tb/8 + k2*tb/8 + j1+24]);
__m256 v0_7 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+7)*nb + i1+0]);
__m256 v0_6 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+6)*nb + i1+0]);
__m256i ws0_6 = _mm256_srli_epi32(w0, 24);
__m256i ws8_6 = _mm256_srli_epi32(w8, 24);
__m256i ws16_6 = _mm256_srli_epi32(w16, 24);
__m256i ws24_6 = _mm256_srli_epi32(w24, 24);
__m256i wsa0_6= _mm256_and_si256(ws0_6, mask);
__m256i wsa8_6= _mm256_and_si256(ws8_6, mask);
__m256i wsa16_6= _mm256_and_si256(ws16_6, mask);
__m256i wsa24_6= _mm256_and_si256(ws24_6, mask);
__m256 l0_6 = _mm256_cvtepi32_ps(wsa0_6);
__m256 l8_6 = _mm256_cvtepi32_ps(wsa8_6);
__m256 l16_6 = _mm256_cvtepi32_ps(wsa16_6);
__m256 l24_6 = _mm256_cvtepi32_ps(wsa24_6);
acc0_0 = _mm256_fmadd_ps(v0_6, l0_6, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_6, l8_6, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_6, l16_6, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_6, l24_6, acc0_24);
__m256i ws0_7 = _mm256_srli_epi32(w0, 28);
__m256i ws8_7 = _mm256_srli_epi32(w8, 28);
__m256i ws16_7 = _mm256_srli_epi32(w16, 28);
__m256i ws24_7 = _mm256_srli_epi32(w24, 28);
__m256i wsa0_7= _mm256_and_si256(ws0_7, mask);
__m256i wsa8_7= _mm256_and_si256(ws8_7, mask);
__m256i wsa16_7= _mm256_and_si256(ws16_7, mask);
__m256i wsa24_7= _mm256_and_si256(ws24_7, mask);
__m256 l0_7 = _mm256_cvtepi32_ps(wsa0_7);
__m256 l8_7 = _mm256_cvtepi32_ps(wsa8_7);
__m256 l16_7 = _mm256_cvtepi32_ps(wsa16_7);
__m256 l24_7 = _mm256_cvtepi32_ps(wsa24_7);
acc0_0 = _mm256_fmadd_ps(v0_7, l0_7, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_7, l8_7, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_7, l16_7, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_7, l24_7, acc0_24);
__m256 v0_5 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+5)*nb + i1+0]);
__m256 v0_4 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+4)*nb + i1+0]);
__m256i ws0_4 = _mm256_srli_epi32(w0, 16);
__m256i ws8_4 = _mm256_srli_epi32(w8, 16);
__m256i ws16_4 = _mm256_srli_epi32(w16, 16);
__m256i ws24_4 = _mm256_srli_epi32(w24, 16);
__m256i wsa0_4= _mm256_and_si256(ws0_4, mask);
__m256i wsa8_4= _mm256_and_si256(ws8_4, mask);
__m256i wsa16_4= _mm256_and_si256(ws16_4, mask);
__m256i wsa24_4= _mm256_and_si256(ws24_4, mask);
__m256 l0_4 = _mm256_cvtepi32_ps(wsa0_4);
__m256 l8_4 = _mm256_cvtepi32_ps(wsa8_4);
__m256 l16_4 = _mm256_cvtepi32_ps(wsa16_4);
__m256 l24_4 = _mm256_cvtepi32_ps(wsa24_4);
acc0_0 = _mm256_fmadd_ps(v0_4, l0_4, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_4, l8_4, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_4, l16_4, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_4, l24_4, acc0_24);
__m256i ws0_5 = _mm256_srli_epi32(w0, 20);
__m256i ws8_5 = _mm256_srli_epi32(w8, 20);
__m256i ws16_5 = _mm256_srli_epi32(w16, 20);
__m256i ws24_5 = _mm256_srli_epi32(w24, 20);
__m256i wsa0_5= _mm256_and_si256(ws0_5, mask);
__m256i wsa8_5= _mm256_and_si256(ws8_5, mask);
__m256i wsa16_5= _mm256_and_si256(ws16_5, mask);
__m256i wsa24_5= _mm256_and_si256(ws24_5, mask);
__m256 l0_5 = _mm256_cvtepi32_ps(wsa0_5);
__m256 l8_5 = _mm256_cvtepi32_ps(wsa8_5);
__m256 l16_5 = _mm256_cvtepi32_ps(wsa16_5);
__m256 l24_5 = _mm256_cvtepi32_ps(wsa24_5);
acc0_0 = _mm256_fmadd_ps(v0_5, l0_5, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_5, l8_5, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_5, l16_5, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_5, l24_5, acc0_24);
__m256 v0_3 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+3)*nb + i1+0]);
__m256 v0_2 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+2)*nb + i1+0]);
__m256i ws0_2 = _mm256_srli_epi32(w0, 8);
__m256i ws8_2 = _mm256_srli_epi32(w8, 8);
__m256i ws16_2 = _mm256_srli_epi32(w16, 8);
__m256i ws24_2 = _mm256_srli_epi32(w24, 8);
__m256i wsa0_2= _mm256_and_si256(ws0_2, mask);
__m256i wsa8_2= _mm256_and_si256(ws8_2, mask);
__m256i wsa16_2= _mm256_and_si256(ws16_2, mask);
__m256i wsa24_2= _mm256_and_si256(ws24_2, mask);
__m256 l0_2 = _mm256_cvtepi32_ps(wsa0_2);
__m256 l8_2 = _mm256_cvtepi32_ps(wsa8_2);
__m256 l16_2 = _mm256_cvtepi32_ps(wsa16_2);
__m256 l24_2 = _mm256_cvtepi32_ps(wsa24_2);
acc0_0 = _mm256_fmadd_ps(v0_2, l0_2, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_2, l8_2, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_2, l16_2, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_2, l24_2, acc0_24);
__m256i ws0_3 = _mm256_srli_epi32(w0, 12);
__m256i ws8_3 = _mm256_srli_epi32(w8, 12);
__m256i ws16_3 = _mm256_srli_epi32(w16, 12);
__m256i ws24_3 = _mm256_srli_epi32(w24, 12);
__m256i wsa0_3= _mm256_and_si256(ws0_3, mask);
__m256i wsa8_3= _mm256_and_si256(ws8_3, mask);
__m256i wsa16_3= _mm256_and_si256(ws16_3, mask);
__m256i wsa24_3= _mm256_and_si256(ws24_3, mask);
__m256 l0_3 = _mm256_cvtepi32_ps(wsa0_3);
__m256 l8_3 = _mm256_cvtepi32_ps(wsa8_3);
__m256 l16_3 = _mm256_cvtepi32_ps(wsa16_3);
__m256 l24_3 = _mm256_cvtepi32_ps(wsa24_3);
acc0_0 = _mm256_fmadd_ps(v0_3, l0_3, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_3, l8_3, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_3, l16_3, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_3, l24_3, acc0_24);
__m256 v0_1 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+1)*nb + i1+0]);
__m256 v0_0 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+0)*nb + i1+0]);
__m256i ws0_0 = _mm256_srli_epi32(w0, 0);
__m256i ws8_0 = _mm256_srli_epi32(w8, 0);
__m256i ws16_0 = _mm256_srli_epi32(w16, 0);
__m256i ws24_0 = _mm256_srli_epi32(w24, 0);
__m256i wsa0_0= _mm256_and_si256(ws0_0, mask);
__m256i wsa8_0= _mm256_and_si256(ws8_0, mask);
__m256i wsa16_0= _mm256_and_si256(ws16_0, mask);
__m256i wsa24_0= _mm256_and_si256(ws24_0, mask);
__m256 l0_0 = _mm256_cvtepi32_ps(wsa0_0);
__m256 l8_0 = _mm256_cvtepi32_ps(wsa8_0);
__m256 l16_0 = _mm256_cvtepi32_ps(wsa16_0);
__m256 l24_0 = _mm256_cvtepi32_ps(wsa24_0);
acc0_0 = _mm256_fmadd_ps(v0_0, l0_0, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_0, l8_0, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_0, l16_0, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_0, l24_0, acc0_24);
__m256i ws0_1 = _mm256_srli_epi32(w0, 4);
__m256i ws8_1 = _mm256_srli_epi32(w8, 4);
__m256i ws16_1 = _mm256_srli_epi32(w16, 4);
__m256i ws24_1 = _mm256_srli_epi32(w24, 4);
__m256i wsa0_1= _mm256_and_si256(ws0_1, mask);
__m256i wsa8_1= _mm256_and_si256(ws8_1, mask);
__m256i wsa16_1= _mm256_and_si256(ws16_1, mask);
__m256i wsa24_1= _mm256_and_si256(ws24_1, mask);
__m256 l0_1 = _mm256_cvtepi32_ps(wsa0_1);
__m256 l8_1 = _mm256_cvtepi32_ps(wsa8_1);
__m256 l16_1 = _mm256_cvtepi32_ps(wsa16_1);
__m256 l24_1 = _mm256_cvtepi32_ps(wsa24_1);
acc0_0 = _mm256_fmadd_ps(v0_1, l0_1, acc0_0);
acc0_8 = _mm256_fmadd_ps(v0_1, l8_1, acc0_8);
acc0_16 = _mm256_fmadd_ps(v0_1, l16_1, acc0_16);
acc0_24 = _mm256_fmadd_ps(v0_1, l24_1, acc0_24);
}
__m256 o0_0 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+0]);
__m256 o0_8 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+8]);
__m256 o0_16 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+16]);
__m256 o0_24 = _mm256_loadu_ps(&output[base_output + j + (i1+0)*t + j1+24]);
__m256 s0_0 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+0]);
__m256 s0_8 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+8]);
__m256 s0_16 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+16]);
__m256 s0_24 = _mm256_loadu_ps(&scales[(k*mb+k1)/gs * t + base_output + j + j1+24]);
__m256 f0_0 = _mm256_fmadd_ps(acc0_0, s0_0, o0_0);
__m256 f0_8 = _mm256_fmadd_ps(acc0_8, s0_8, o0_8);
__m256 f0_16 = _mm256_fmadd_ps(acc0_16, s0_16, o0_16);
__m256 f0_24 = _mm256_fmadd_ps(acc0_24, s0_24, o0_24);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+0], f0_0);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+8], f0_8);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+16], f0_16);
_mm256_storeu_ps(&output[base_output + j + (i1+0)*t + j1+24], f0_24);
}
}
}
}
}
}
#pragma omp barrier
const int ngs = m/gs;
for (int i = 0; i < n; i++) {
for (int j = 0; j < tt; j+=32){
__m256 acc0 = _mm256_setzero_ps();
__m256 acc8 = _mm256_setzero_ps();
__m256 acc16 = _mm256_setzero_ps();
__m256 acc24 = _mm256_setzero_ps();
for (int i1 = 0; i1 < ngs; i1++){
__m256 r = _mm256_set1_ps(sums[i*ngs + i1]);
__m256 z0 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 0]);
__m256 z8 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 8]);
__m256 z16 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 16]);
__m256 z24 = _mm256_loadu_ps(&zeros[base_output + i1* t + j + 24]);
__m256 s0 = _mm256_loadu_ps(&scales[base_output + i1 * t + j + 0]);
__m256 s8 = _mm256_loadu_ps(&scales[base_output + i1 * t + j + 8]);
__m256 s16 = _mm256_loadu_ps(&scales[base_output + i1 * t + j + 16]);
__m256 s24 = _mm256_loadu_ps(&scales[base_output + i1 * t + j + 24]);
__m256 zs0 = _mm256_mul_ps(z0, s0);
__m256 zs8 = _mm256_mul_ps(z8, s8);
__m256 zs16 = _mm256_mul_ps(z16, s16);
__m256 zs24 = _mm256_mul_ps(z24, s24);
acc0 = _mm256_fmadd_ps(zs0, r, acc0);
acc8 = _mm256_fmadd_ps(zs8, r, acc8);
acc16 = _mm256_fmadd_ps(zs16, r, acc16);
acc24 = _mm256_fmadd_ps(zs24, r, acc24);
}
__m256 o0 = _mm256_loadu_ps(&output[i*t + base_output + j + 0]);
__m256 o8 = _mm256_loadu_ps(&output[i*t + base_output + j + 8]);
__m256 o16 = _mm256_loadu_ps(&output[i*t + base_output + j + 16]);
__m256 o24 = _mm256_loadu_ps(&output[i*t + base_output + j + 24]);
__m256 b0 = _mm256_loadu_ps(&bias[base_output + j + 0]);
__m256 b8 = _mm256_loadu_ps(&bias[base_output + j + 8]);
__m256 b16 = _mm256_loadu_ps(&bias[base_output + j + 16]);
__m256 b24 = _mm256_loadu_ps(&bias[base_output + j + 24]);
__m256 o10 = _mm256_sub_ps(o0, acc0);
__m256 o18 = _mm256_sub_ps(o8, acc8);
__m256 o116 = _mm256_sub_ps(o16, acc16);
__m256 o124 = _mm256_sub_ps(o24, acc24);
__m256 o20 = _mm256_add_ps(o10, b0);
__m256 o28 = _mm256_add_ps(o18, b8);
__m256 o216 = _mm256_add_ps(o116, b16);
__m256 o224 = _mm256_add_ps(o124, b24);
_mm256_storeu_ps(&output[i*t + base_output + j + 0], o20);
_mm256_storeu_ps(&output[i*t + base_output + j + 8], o28);
_mm256_storeu_ps(&output[i*t + base_output + j + 16], o216);
_mm256_storeu_ps(&output[i*t + base_output + j + 24], o224);
}
}
}
}
inline void forward4_gs_cpu(
torch::Tensor in, torch::Tensor weight, torch::Tensor out,
torch::Tensor bias, torch::Tensor scales, torch::Tensor zeros, torch::Tensor sums,
int N, int M, int T, int nb, int mb, int tb, int tt, int groupsize, int cutoff){
int*   W = weight.data_ptr<int>();
float* input = in.data_ptr<float>();
float* b   = bias.data_ptr<float>();
float* s   = scales.data_ptr<float>();
float* z   = zeros.data_ptr<float>();
float* r   = sums.data_ptr<float>();
float* O   = out.data_ptr<float>();

q4gemm_gs(input, W, s, z, b, r, O, N, M, T, nb, mb, tb, tt, groupsize, cutoff);
}
inline void pack4_qw_inner(int* A, int* B, const int N, const int M, const int nb, int mb, int cutoff){
// copy the full matrix A in blocked format into B
uint64_t idx = 0;
for(int j = 0, tid = 0; j < M; j+=mb, tid++){
for(int i = 0; i < N; i+=nb){
 for(int ii = i; ii < mymin(i+nb, N); ii++){
 for(int jj = j; jj < mymin(j+mb, M); jj++){
 B[idx] = A[ii*M+jj];
 idx++;
 }
 }
 }
}
}
inline void pack4_w_cpu(
torch::Tensor in, torch::Tensor out,
int N, int M, int nb, int mb, int cutoff){
int* input = in.data_ptr<int>();
int* O = out.data_ptr<int>();
  pack4_qw_inner(input, O, N, M, nb, mb, cutoff);
}
void unpack_zeros4_cpu(const int* zv, float* ov, int n, int m){
const __m256i ones = _mm256_set1_epi32(1);
const __m256i mask = _mm256_set1_epi32(15);
const __m256i shift = _mm256_set_epi32(28,24,20,16,12,8,4,0);
for(int i = 0; i < n; i++){
for(int j = 0; j < m; j+=8){
__m256i z = _mm256_set1_epi32(zv[i*m/8 + j/8]);
__m256i z0 = _mm256_srlv_epi32(z, shift);
__m256i z1 = _mm256_and_si256(z0, mask);
__m256i z2 = _mm256_add_epi32(z1, ones);
__m256 z3 = _mm256_cvtepi32_ps(z2);
_mm256_storeu_ps(&ov[i*m +j], z3);
}
}
}
void unpack_zeros4(torch::Tensor zeros, torch::Tensor out, int N, int M){
int* Z = zeros.data_ptr<int>();
float* O = out.data_ptr<float>();
unpack_zeros4_cpu(Z, O, N, M);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward2", &forward2_cpu);
  m.def("forward3", &forward3_cpu);
  m.def("forward4", &forward4_cpu);
  m.def("unpack_zeros2", &unpack_zeros2);
  m.def("unpack_zeros3", &unpack_zeros3);
  m.def("unpack_zeros4", &unpack_zeros4);
  m.def("forward_gs2", &forward2_gs_cpu);
  m.def("forward_gs3", &forward3_gs_cpu);
  m.def("forward_gs4", &forward4_gs_cpu);
  m.def("pack2", &pack2_w_cpu);
  m.def("pack3", &pack3_w_cpu);
  m.def("pack4", &pack4_w_cpu);
m.def("compute_reduction_cpp", &compute_reduction);
m.def("unquantize_sim", &unquantize_sim);
m.def("quant_scalar_scaled", &quant_scalar_cpu);
}

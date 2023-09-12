#include<omp.h>
#include<immintrin.h>
#include<fstream>

#define mymin(a,b) ((a)<(b)?(a):(b))
#define mymax(a,b) ((a)>(b)?(a):(b))
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
__m256 v0_14 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+14)*nb + i1+0]);
__m256 v0_13 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+13)*nb + i1+0]);
__m256 v0_12 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+12)*nb + i1+0]);
__m256 v0_11 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+11)*nb + i1+0]);
__m256 v0_10 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+10)*nb + i1+0]);
__m256 v0_9 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+9)*nb + i1+0]);
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
__m256 v0_7 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+7)*nb + i1+0]);
__m256 v0_6 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+6)*nb + i1+0]);
__m256 v0_5 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+5)*nb + i1+0]);
__m256 v0_4 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+4)*nb + i1+0]);
__m256 v0_3 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+3)*nb + i1+0]);
__m256 v0_2 = _mm256_set1_ps(input[(i*om+k)*mb*nb + (k2+2)*nb + i1+0]);
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
__m256 o10 = _mm256_add_ps(o0, acc0);
__m256 o18 = _mm256_add_ps(o8, acc8);
__m256 o116 = _mm256_add_ps(o16, acc16);
__m256 o124 = _mm256_add_ps(o24, acc24);
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
inline void qforward(const float* __restrict__ input, 
 const int* __restrict__ W, 
const float* __restrict__ scales, 
const float* __restrict__ zeros, 
const float* __restrict__ bias, 
const float* __restrict__ sums, 
float* __restrict__ output, 
int n, 
 int m, 
 int t) {
q2gemm_gs(input, W, scales, zeros, bias, sums, output, n, m, t, 1, 1024, 32, 352, 64, 8);
}
inline void pack_input(float* A, float* B){
  // copy the full matrix A in blocked format into B
  uint64_t idx = 0;
  const int N = 1;
  const int M = 4096;
  const int nb = 1;
  const int mb = 1024;
  for(int i = 0; i < N; i+=nb){ 
             for(int j = 0; j < M; j+=mb){
                 for(int jj = j; jj < mymin(j+mb, M); jj++){
                     for(int ii = i; ii < mymin(i+nb, N); ii++){
                         B[idx] = A[ii*M+jj];
                         idx++;
                     }
                 }
             }
         }
     }
inline void pack_qw_inner(int* A, int* B, int cutoff){
  // copy the full matrix A in blocked format into B
  uint64_t idx = 0;
  const int N = 256;
  const int M = 4096;
  const int nb = 64;
int mb = 32;
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
inline void pack_qw(int* A, int* B){
  pack_qw_inner(A, B, 65);
}
inline void pack_output(float* A, float* B){
  // copy the full matrix A in blocked format into B
  uint64_t idx = 0;
  const int N = 1;
  const int M = 4096;
  const int nb = 1;
  const int mb = 32;
  for(int i = 0; i < N; i+=nb){ 
             for(int j = 0; j < M; j+=mb){
                 for(int ii = i; ii < mymin(i+nb, N); ii++){
                     for(int jj = j; jj < mymin(j+mb, M); jj++){
                         B[idx] = A[ii*M+jj];
                         idx++;
                     }
                 }
             }
         }
     }
void print_parameters(){
std::ofstream outfile;
outfile.open("./autogptq_extension/qigen/tmp.csv", std::ios_base::app);
outfile << 2 << "," << 1 << "," << 16 << "," << 32 << "," << 8 << "," << 12  << "," << 64 << ",";
}
